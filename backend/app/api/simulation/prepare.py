"""Create and prepare simulations."""

import traceback
from flask import request, jsonify

from .. import simulation_bp
from ...services import option_compare
from ...services.entity_reader import EntityReader
from ...services.simulation_manager import SimulationManager, SimulationStatus
from ... import run_mode
from ...utils.logger import get_logger
from ...utils.locale import t, get_locale, set_locale
from ...models.project import ProjectManager

logger = get_logger('mirofish.api.simulation')
from ._shared import (
    _check_simulation_prepared,
    _model_service_problem,
)


# ============== 模拟管理接口 ==============

@simulation_bp.route('/create', methods=['POST'])
def create_simulation():
    """
    创建新的模拟
    
    注意：max_rounds等参数由LLM智能生成，无需手动设置
    
    请求（JSON）：
        {
            "project_id": "proj_xxxx",      // 必填
            "graph_id": "mirofish_xxxx",    // 可选，如不提供则从project获取
            "enable_twitter": true,          // 可选，默认true
            "enable_reddit": true            // 可选，默认true
        }
    
    返回：
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "project_id": "proj_xxxx",
                "graph_id": "mirofish_xxxx",
                "status": "created",
                "enable_twitter": true,
                "enable_reddit": true,
                "created_at": "2025-12-01T10:00:00"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        project_id = data.get('project_id')
        if not project_id:
            return jsonify({
                "success": False,
                "error": t('api.requireProjectId')
            }), 400
        
        project = ProjectManager.get_project(project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": t('api.projectNotFound', id=project_id)
            }), 404
        
        graph_id = data.get('graph_id') or project.graph_id
        if not graph_id:
            return jsonify({
                "success": False,
                "error": t('api.graphNotBuilt')
            }), 400
        
        manager = SimulationManager()
        state = manager.create_simulation(
            project_id=project_id,
            graph_id=graph_id,
            enable_twitter=data.get('enable_twitter', True),
            enable_reddit=data.get('enable_reddit', True),
        )
        
        return jsonify({
            "success": True,
            "data": state.to_dict()
        })
        
    except Exception as e:
        logger.error(f"创建模拟失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/copy', methods=['POST'])
def copy_simulation():
    """A ready copy of a prepared simulation, for another seed (#63).

    JSON: {"simulation_id": "sim_xxxx"} -> {"simulation_id": <the copy>}
    """
    data = request.get_json(silent=True) or {}
    source_id = data.get('simulation_id')
    if not isinstance(source_id, str) or not source_id:
        return jsonify({"success": False, "error": t('api.requireSimulationId')}), 400
    try:
        state = SimulationManager().copy_simulation(source_id)
    except ValueError as error:
        return jsonify({"success": False, "error": str(error)}), 404 if "not found" in str(error) else 409
    except OSError as error:
        logger.error(f"copy of {source_id} failed: {error}")
        return jsonify({"success": False, "error": f"copy failed: {error}"}), 500
    return jsonify({"success": True, "data": state.to_dict()})


@simulation_bp.route('/prepare', methods=['POST'])
def prepare_simulation():
    """
    准备模拟环境（异步任务，LLM智能生成所有参数）
    
    这是一个耗时操作，接口会立即返回task_id，
    使用 GET /api/simulation/prepare/status 查询进度
    
    特性：
    - 自动检测已完成的准备工作，避免重复生成
    - 如果已准备完成，直接返回已有结果
    - 支持强制重新生成（force_regenerate=true）
    
    步骤：
    1. 检查是否已有完成的准备工作
    2. 从Zep图谱读取并过滤实体
    3. 为每个实体生成OASIS Agent Profile（带重试机制）
    4. LLM智能生成模拟配置（带重试机制）
    5. 保存配置文件和预设脚本
    
    请求（JSON）：
        {
            "simulation_id": "sim_xxxx",                   // 必填，模拟ID
            "entity_types": ["Student", "PublicFigure"],  // 可选，指定实体类型
            "use_llm_for_profiles": true,                 // 可选，是否用LLM生成人设
            "parallel_profile_count": 5,                  // 可选，并行生成人设数量，默认5
            "force_regenerate": false                     // 可选，强制重新生成，默认false
        }
    
    返回：
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "task_id": "task_xxxx",           // 新任务时返回
                "status": "preparing|ready",
                "message": "准备任务已启动|已有完成的准备工作",
                "already_prepared": true|false    // 是否已准备完成
            }
        }
    """
    problem = _model_service_problem()
    if problem:
        return jsonify({"success": False, "error": problem}), 503
    import threading
    from ...models.task import TaskManager, TaskStatus
    
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": t('api.requireSimulationId')
            }), 400
        
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        
        if not state:
            return jsonify({
                "success": False,
                "error": t('api.simulationNotFound', id=simulation_id)
            }), 404
        
        # 检查是否强制重新生成
        force_regenerate = data.get('force_regenerate', False)
        logger.info(f"开始处理 /prepare 请求: simulation_id={simulation_id}, force_regenerate={force_regenerate}")
        
        # 检查是否已经准备完成（避免重复生成）
        if not force_regenerate:
            logger.debug(f"检查模拟 {simulation_id} 是否已准备完成...")
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
            logger.debug(f"检查结果: is_prepared={is_prepared}, prepare_info={prepare_info}")
            if is_prepared:
                logger.info(f"模拟 {simulation_id} 已准备完成，跳过重复生成")
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "ready",
                        "message": t('api.alreadyPrepared'),
                        "already_prepared": True,
                        "prepare_info": prepare_info
                    }
                })
            else:
                logger.info(f"模拟 {simulation_id} 未准备完成，将启动准备任务")
        
        # 从项目获取必要信息
        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": t('api.projectNotFound', id=state.project_id)
            }), 404
        
        # 获取模拟需求
        simulation_requirement = project.simulation_requirement or ""
        if not simulation_requirement:
            return jsonify({
                "success": False,
                "error": t('api.projectMissingRequirement')
            }), 400
        
        # 获取文档文本
        document_text = ProjectManager.get_extracted_text(state.project_id) or ""

        # One option of a comparison (#56, #66): announced in the requirement
        # and the document, on the project's shared graph.
        if data.get('option') is not None:
            try:
                option = option_compare.validate_one(data.get('option'))
            except ValueError as error:
                return jsonify({"success": False, "error": str(error)}), 400
            simulation_requirement, document_text = option_compare.with_option(
                simulation_requirement, document_text, option)
        
        entity_types_list = data.get('entity_types')
        use_llm_for_profiles = data.get('use_llm_for_profiles', True)
        parallel_profile_count = data.get('parallel_profile_count', 5)
        
        # ========== 同步获取实体数量（在后台任务启动前） ==========
        # 这样前端在调用prepare后立即就能获取到预期Agent总数
        try:
            logger.info(f"同步获取实体数量: graph_id={state.graph_id}")
            reader = EntityReader()
            # 快速读取实体（不需要边信息，只统计数量）
            filtered_preview = reader.filter_defined_entities(
                graph_id=state.graph_id,
                defined_entity_types=entity_types_list,
                enrich_with_edges=False  # 不获取边信息，加快速度
            )
            # 保存实体数量到状态（供前端立即获取）
            state.entities_count = filtered_preview.filtered_count
            state.entity_types = list(filtered_preview.entity_types)
            logger.info(f"预期实体数量: {filtered_preview.filtered_count}, 类型: {filtered_preview.entity_types}")
        except Exception as e:
            logger.warning(f"同步获取实体数量失败（将在后台任务中重试）: {e}")
            # 失败不影响后续流程，后台任务会重新获取
        
        # 创建异步任务
        task_manager = TaskManager()
        task_id = task_manager.create_task(
            task_type="simulation_prepare",
            metadata={
                "simulation_id": simulation_id,
                "project_id": state.project_id
            }
        )
        
        # 更新模拟状态（包含预先获取的实体数量）
        state.status = SimulationStatus.PREPARING
        manager._save_simulation_state(state)
        
        # Capture locale before spawning background thread
        current_locale = get_locale()

        # 定义后台任务
        def run_prepare():
            set_locale(current_locale)
            try:
                task_manager.update_task(
                    task_id,
                    status=TaskStatus.PROCESSING,
                    progress=0,
                    message=t('progress.startPreparingEnv')
                )
                
                # 准备模拟（带进度回调）
                # 存储阶段进度详情
                stage_details = {}
                
                def progress_callback(stage, progress, message, **kwargs):
                    # 计算总进度
                    stage_weights = {
                        "reading": (0, 20),           # 0-20%
                        "generating_profiles": (20, 70),  # 20-70%
                        "generating_config": (70, 90),    # 70-90%
                        "copying_scripts": (90, 100)       # 90-100%
                    }
                    
                    start, end = stage_weights.get(stage, (0, 100))
                    current_progress = int(start + (end - start) * progress / 100)
                    
                    # 构建详细进度信息
                    stage_names = {
                        "reading": t('progress.readingGraphEntities'),
                        "generating_profiles": t('progress.generatingProfiles'),
                        "generating_config": t('progress.generatingSimConfig'),
                        "copying_scripts": t('progress.preparingScripts')
                    }
                    
                    stage_index = list(stage_weights.keys()).index(stage) + 1 if stage in stage_weights else 1
                    total_stages = len(stage_weights)
                    
                    # 更新阶段详情
                    stage_details[stage] = {
                        "stage_name": stage_names.get(stage, stage),
                        "stage_progress": progress,
                        "current": kwargs.get("current", 0),
                        "total": kwargs.get("total", 0),
                        "item_name": kwargs.get("item_name", "")
                    }
                    
                    # 构建详细进度信息
                    detail = stage_details[stage]
                    progress_detail_data = {
                        "current_stage": stage,
                        "current_stage_name": stage_names.get(stage, stage),
                        "stage_index": stage_index,
                        "total_stages": total_stages,
                        "stage_progress": progress,
                        "current_item": detail["current"],
                        "total_items": detail["total"],
                        "item_description": message
                    }
                    
                    # 构建简洁消息
                    if detail["total"] > 0:
                        detailed_message = (
                            f"[{stage_index}/{total_stages}] {stage_names.get(stage, stage)}: "
                            f"{detail['current']}/{detail['total']} - {message}"
                        )
                    else:
                        detailed_message = f"[{stage_index}/{total_stages}] {stage_names.get(stage, stage)}: {message}"
                    
                    task_manager.update_task(
                        task_id,
                        progress=current_progress,
                        message=detailed_message,
                        progress_detail=progress_detail_data
                    )
                
                result_state = manager.prepare_simulation(
                    simulation_id=simulation_id,
                    simulation_requirement=simulation_requirement,
                    document_text=document_text,
                    defined_entity_types=entity_types_list,
                    use_llm_for_profiles=use_llm_for_profiles,
                    progress_callback=progress_callback,
                    parallel_profile_count=parallel_profile_count
                )

                if result_state.status == SimulationStatus.FAILED:
                    task_manager.fail_task(
                        task_id,
                        result_state.error or "模拟准备失败"
                    )
                else:
                    task_manager.complete_task(
                        task_id,
                        result=result_state.to_simple_dict()
                    )
                
            except Exception as e:
                logger.error(f"准备模拟失败: {str(e)}")
                task_manager.fail_task(task_id, str(e))
                
                # 更新模拟状态为失败
                state = manager.get_simulation(simulation_id)
                if state:
                    state.status = SimulationStatus.FAILED
                    state.error = str(e)
                    manager._save_simulation_state(state)
        
        # 启动后台线程
        thread = threading.Thread(target=run_mode.bind(run_prepare), daemon=True)  # keeps the run's mode (#66)
        thread.start()
        
        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "task_id": task_id,
                "status": "preparing",
                "message": t('api.prepareStarted'),
                "already_prepared": False,
                "expected_entities_count": state.entities_count,  # 预期的Agent总数
                "entity_types": state.entity_types  # 实体类型列表
            }
        })
        
    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 404
        
    except Exception as e:
        logger.error(f"启动准备任务失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@simulation_bp.route('/prepare/status', methods=['POST'])
def get_prepare_status():
    """
    查询准备任务进度
    
    支持两种查询方式：
    1. 通过task_id查询正在进行的任务进度
    2. 通过simulation_id检查是否已有完成的准备工作
    
    请求（JSON）：
        {
            "task_id": "task_xxxx",          // 可选，prepare返回的task_id
            "simulation_id": "sim_xxxx"      // 可选，模拟ID（用于检查已完成的准备）
        }
    
    返回：
        {
            "success": true,
            "data": {
                "task_id": "task_xxxx",
                "status": "processing|completed|ready",
                "progress": 45,
                "message": "...",
                "already_prepared": true|false,  // 是否已有完成的准备
                "prepare_info": {...}            // 已准备完成时的详细信息
            }
        }
    """
    from ...models.task import TaskManager
    
    try:
        data = request.get_json() or {}
        
        task_id = data.get('task_id')
        simulation_id = data.get('simulation_id')
        
        # 如果提供了simulation_id，先检查是否已准备完成
        if simulation_id:
            is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
            if is_prepared:
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "ready",
                        "progress": 100,
                        "message": t('api.alreadyPrepared'),
                        "already_prepared": True,
                        "prepare_info": prepare_info
                    }
                })
        
        # 如果没有task_id，返回错误
        if not task_id:
            if simulation_id:
                # 有simulation_id但未准备完成
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "status": "not_started",
                        "progress": 0,
                        "message": t('api.notStartedPrepare'),
                        "already_prepared": False
                    }
                })
            return jsonify({
                "success": False,
                "error": t('api.requireTaskOrSimId')
            }), 400
        
        task_manager = TaskManager()
        task = task_manager.get_task(task_id)
        
        if not task:
            # 任务不存在，但如果有simulation_id，检查是否已准备完成
            if simulation_id:
                is_prepared, prepare_info = _check_simulation_prepared(simulation_id)
                if is_prepared:
                    return jsonify({
                        "success": True,
                        "data": {
                            "simulation_id": simulation_id,
                            "task_id": task_id,
                            "status": "ready",
                            "progress": 100,
                            "message": t('api.taskCompletedPrepared'),
                            "already_prepared": True,
                            "prepare_info": prepare_info
                        }
                    })
            
            return jsonify({
                "success": False,
                "error": t('api.taskNotFound', id=task_id)
            }), 404
        
        task_dict = task.to_dict()
        task_dict["already_prepared"] = False
        
        return jsonify({
            "success": True,
            "data": task_dict
        })
        
    except Exception as e:
        logger.error(f"查询任务状态失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
