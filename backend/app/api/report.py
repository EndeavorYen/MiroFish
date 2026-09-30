"""
Report API路由
提供模拟报告生成、获取、对话等接口
"""

import os
import traceback
import threading
from flask import request, jsonify, send_file

from . import report_bp
from ..config import Config
from ..services.report_agent import ReportAgent, ReportManager, ReportStatus
from ..services.simulation_manager import SimulationManager
from ..services.simulation_runner import SimulationRunner, RunnerStatus
from ..services.graph_memory_updater import GraphMemoryManager
from ..models.project import ProjectManager, ProjectStatus
from ..models.task import TaskManager, TaskStatus
from ..utils.logger import get_logger
from ..utils.locale import t, get_locale, set_locale
from ..utils.graph_lifecycle import (
    graph_lifecycle_lock,
    register_graph_reader,
    unregister_graph_reader,
)

logger = get_logger('mirofish.api.report')


# ============== 报告生成接口 ==============

@report_bp.route('/generate', methods=['POST'])
def generate_report():
    """
    生成模拟分析报告（异步任务）
    
    这是一个耗时操作，接口会立即返回task_id，
    使用 GET /api/report/generate/status 查询进度
    
    请求（JSON）：
        {
            "simulation_id": "sim_xxxx",    // 必填，模拟ID
            "force_regenerate": false        // 可选，强制重新生成
        }
    
    返回：
        {
            "success": true,
            "data": {
                "simulation_id": "sim_xxxx",
                "task_id": "task_xxxx",
                "status": "generating",
                "message": "报告生成任务已启动"
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": t('api.requireSimulationId')
            }), 400

        force_regenerate = data.get('force_regenerate', False)
        if not isinstance(force_regenerate, bool):
            return jsonify({
                "success": False,
                "error": "force_regenerate must be a JSON boolean",
            }), 400
        
        # 获取模拟信息
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        
        if not state:
            return jsonify({
                "success": False,
                "error": t('api.simulationNotFound', id=simulation_id)
            }), 404

        run_state = SimulationRunner.get_run_state(simulation_id)
        updater = GraphMemoryManager.get_updater(simulation_id)
        active_statuses = {
            RunnerStatus.STARTING,
            RunnerStatus.RUNNING,
            RunnerStatus.PAUSED,
            RunnerStatus.STOPPING,
        }
        if updater is not None or (
            run_state is not None and run_state.runner_status in active_statuses
        ):
            return jsonify({
                "success": False,
                "error": (
                    "Simulation or Zep graph ingestion is still active; "
                    "wait for a terminal run status before generating a report"
                ),
                "ingestion_pending": updater is not None,
            }), 409
        successful_terminal_statuses = {
            RunnerStatus.COMPLETED,
            RunnerStatus.STOPPED,
        }
        if (
            run_state is None
            or run_state.runner_status not in successful_terminal_statuses
        ):
            return jsonify({
                "success": False,
                "error": (
                    "A successfully completed or stopped simulation is required "
                    "before generating a report"
                ),
            }), 409

        # 获取项目信息
        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": t('api.projectNotFound', id=state.project_id)
            }), 404
        
        if project.status != ProjectStatus.GRAPH_COMPLETED:
            return jsonify({
                "success": False,
                "error": "The project graph must be completely built before reporting",
            }), 409

        graph_id = project.graph_id
        if not graph_id:
            return jsonify({
                "success": False,
                "error": t('api.missingGraphIdEnsure')
            }), 400
        if state.graph_id and state.graph_id != graph_id:
            return jsonify({
                "success": False,
                "error": (
                    "The simulation references an older graph; prepare it "
                    "again before generating a report"
                ),
            }), 409
        
        simulation_requirement = project.simulation_requirement
        if not simulation_requirement:
            return jsonify({
                "success": False,
                "error": t('api.missingSimRequirement')
            }), 400
        
        # 提前生成 report_id，以便立即返回给前端
        import uuid
        report_id = f"report_{uuid.uuid4().hex[:12]}"
        
        # Register the background report as a graph reader under the same lock
        # used by graph deletion and updater startup. A lock itself cannot be
        # acquired in this request thread and released by the worker, so the
        # durable reader registration is the cross-thread lease.
        with graph_lifecycle_lock(graph_id):
            refreshed_state = manager.get_simulation(simulation_id)
            refreshed_project = (
                ProjectManager.get_project(refreshed_state.project_id)
                if refreshed_state
                else None
            )
            refreshed_run_state = SimulationRunner.get_run_state(simulation_id)
            refreshed_updater = GraphMemoryManager.get_updater(simulation_id)
            if (
                refreshed_state is None
                or refreshed_project is None
                or refreshed_project.graph_id != graph_id
                or refreshed_project.status != ProjectStatus.GRAPH_COMPLETED
                or (
                    refreshed_state.graph_id
                    and refreshed_state.graph_id != graph_id
                )
            ):
                return jsonify({
                    "success": False,
                    "error": "The project graph changed while reporting was starting",
                }), 409
            if refreshed_updater is not None or (
                refreshed_run_state is not None
                and refreshed_run_state.runner_status in active_statuses
            ):
                return jsonify({
                    "success": False,
                    "error": (
                        "Simulation or Zep graph ingestion became active; "
                        "retry after it reaches a terminal state"
                    ),
                    "ingestion_pending": refreshed_updater is not None,
                }), 409
            if (
                refreshed_run_state is None
                or refreshed_run_state.runner_status
                not in successful_terminal_statuses
            ):
                return jsonify({
                    "success": False,
                    "error": (
                        "A successfully completed or stopped simulation is "
                        "required before generating a report"
                    ),
                }), 409

            # Cached-report reuse is now part of the same atomic barrier, so a
            # concurrent rerun cannot make the returned report stale between
            # the status check and response.
            if not force_regenerate:
                existing_report = ReportManager.get_report_by_simulation(
                    simulation_id
                )
                if (
                    existing_report
                    and existing_report.status == ReportStatus.COMPLETED
                ):
                    return jsonify({
                        "success": True,
                        "data": {
                            "simulation_id": simulation_id,
                            "report_id": existing_report.report_id,
                            "status": "completed",
                            "message": t('api.reportAlreadyExists'),
                            "already_generated": True
                        }
                    })

            task_manager = TaskManager()
            task_id = task_manager.create_task(
                task_type="report_generate",
                metadata={
                    "simulation_id": simulation_id,
                    "graph_id": graph_id,
                    "report_id": report_id
                }
            )
            current_locale = get_locale()
            register_graph_reader(graph_id, report_id)

            def run_generate():
                set_locale(current_locale)
                try:
                    task_manager.update_task(
                        task_id,
                        status=TaskStatus.PROCESSING,
                        progress=0,
                        message=t('api.initReportAgent')
                    )

                    if Config.REPORT_MODE not in ("metrics", "agent"):
                        logger.warning(
                            "unknown REPORT_MODE %r; using the ReportAgent", Config.REPORT_MODE
                        )
                    if Config.REPORT_MODE == "metrics":
                        # Deterministic metrics + one short summary (#12).
                        from ..services.metrics_report import generate_metrics_report

                        report = generate_metrics_report(
                            simulation_id, graph_id, simulation_requirement, report_id
                        )
                        if report.status == ReportStatus.COMPLETED:
                            task_manager.complete_task(
                                task_id,
                                result={
                                    "report_id": report.report_id,
                                    "simulation_id": simulation_id,
                                    "status": "completed",
                                },
                            )
                        else:
                            task_manager.fail_task(task_id, report.error or t('api.reportGenerateFailed'))
                        return

                    agent = ReportAgent(
                        graph_id=graph_id,
                        simulation_id=simulation_id,
                        simulation_requirement=simulation_requirement
                    )

                    def progress_callback(stage, progress, message):
                        task_manager.update_task(
                            task_id,
                            progress=progress,
                            message=f"[{stage}] {message}"
                        )

                    report = agent.generate_report(
                        progress_callback=progress_callback,
                        report_id=report_id
                    )
                    ReportManager.save_report(report)

                    if report.status == ReportStatus.COMPLETED:
                        task_manager.complete_task(
                            task_id,
                            result={
                                "report_id": report.report_id,
                                "simulation_id": simulation_id,
                                "status": "completed"
                            }
                        )
                    else:
                        task_manager.fail_task(
                            task_id,
                            report.error or t('api.reportGenerateFailed')
                        )
                except Exception as e:
                    logger.error(f"报告生成失败: {str(e)}")
                    task_manager.fail_task(task_id, str(e))
                finally:
                    unregister_graph_reader(graph_id, report_id)

            try:
                thread = threading.Thread(target=run_generate, daemon=True)
                thread.start()
            except Exception:
                unregister_graph_reader(graph_id, report_id)
                raise
        
        return jsonify({
            "success": True,
            "data": {
                "simulation_id": simulation_id,
                "report_id": report_id,
                "task_id": task_id,
                "status": "generating",
                "message": t('api.reportGenerateStarted'),
                "already_generated": False
            }
        })
        
    except Exception as e:
        logger.error(f"启动报告生成任务失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@report_bp.route('/generate/status', methods=['POST'])
def get_generate_status():
    """
    查询报告生成任务进度
    
    请求（JSON）：
        {
            "task_id": "task_xxxx",         // 可选，generate返回的task_id
            "simulation_id": "sim_xxxx"     // 可选，模拟ID
        }
    
    返回：
        {
            "success": true,
            "data": {
                "task_id": "task_xxxx",
                "status": "processing|completed|failed",
                "progress": 45,
                "message": "..."
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        task_id = data.get('task_id')
        simulation_id = data.get('simulation_id')
        
        # 如果提供了simulation_id，先检查是否已有完成的报告
        if simulation_id:
            existing_report = ReportManager.get_report_by_simulation(simulation_id)
            if existing_report and existing_report.status == ReportStatus.COMPLETED:
                return jsonify({
                    "success": True,
                    "data": {
                        "simulation_id": simulation_id,
                        "report_id": existing_report.report_id,
                        "status": "completed",
                        "progress": 100,
                        "message": t('api.reportGenerated'),
                        "already_completed": True
                    }
                })
        
        if not task_id:
            return jsonify({
                "success": False,
                "error": t('api.requireTaskOrSimId')
            }), 400
        
        task_manager = TaskManager()
        task = task_manager.get_task(task_id)
        
        if not task:
            return jsonify({
                "success": False,
                "error": t('api.taskNotFound', id=task_id)
            }), 404
        
        return jsonify({
            "success": True,
            "data": task.to_dict()
        })
        
    except Exception as e:
        logger.error(f"查询任务状态失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ============== 报告获取接口 ==============

@report_bp.route('/<report_id>', methods=['GET'])
def get_report(report_id: str):
    """
    获取报告详情
    
    返回：
        {
            "success": true,
            "data": {
                "report_id": "report_xxxx",
                "simulation_id": "sim_xxxx",
                "status": "completed",
                "outline": {...},
                "markdown_content": "...",
                "created_at": "...",
                "completed_at": "..."
            }
        }
    """
    try:
        report = ReportManager.get_report(report_id)
        
        if not report:
            return jsonify({
                "success": False,
                "error": t('api.reportNotFound', id=report_id)
            }), 404
        
        return jsonify({
            "success": True,
            "data": report.to_dict()
        })
        
    except Exception as e:
        logger.error(f"获取报告失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@report_bp.route('/by-simulation/<simulation_id>', methods=['GET'])
def get_report_by_simulation(simulation_id: str):
    """
    根据模拟ID获取报告
    
    返回：
        {
            "success": true,
            "data": {
                "report_id": "report_xxxx",
                ...
            }
        }
    """
    try:
        report = ReportManager.get_report_by_simulation(simulation_id)
        
        if not report:
            return jsonify({
                "success": False,
                "error": t('api.noReportForSim', id=simulation_id),
                "has_report": False
            }), 404
        
        return jsonify({
            "success": True,
            "data": report.to_dict(),
            "has_report": True
        })
        
    except Exception as e:
        logger.error(f"获取报告失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@report_bp.route('/list', methods=['GET'])
def list_reports():
    """
    列出所有报告
    
    Query参数：
        simulation_id: 按模拟ID过滤（可选）
        limit: 返回数量限制（默认50）
    
    返回：
        {
            "success": true,
            "data": [...],
            "count": 10
        }
    """
    try:
        simulation_id = request.args.get('simulation_id')
        limit = request.args.get('limit', 50, type=int)
        
        reports = ReportManager.list_reports(
            simulation_id=simulation_id,
            limit=limit
        )
        
        return jsonify({
            "success": True,
            "data": [r.to_dict() for r in reports],
            "count": len(reports)
        })
        
    except Exception as e:
        logger.error(f"列出报告失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@report_bp.route('/<report_id>/download', methods=['GET'])
def download_report(report_id: str):
    """
    下载报告（Markdown格式）
    
    返回Markdown文件
    """
    try:
        report = ReportManager.get_report(report_id)
        
        if not report:
            return jsonify({
                "success": False,
                "error": t('api.reportNotFound', id=report_id)
            }), 404
        
        md_path = ReportManager._get_report_markdown_path(report_id)
        
        if not os.path.exists(md_path):
            # 如果MD文件不存在，生成一个临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
                f.write(report.markdown_content)
                temp_path = f.name
            
            return send_file(
                temp_path,
                as_attachment=True,
                download_name=f"{report_id}.md"
            )
        
        return send_file(
            md_path,
            as_attachment=True,
            download_name=f"{report_id}.md"
        )
        
    except Exception as e:
        logger.error(f"下载报告失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@report_bp.route('/<report_id>', methods=['DELETE'])
def delete_report(report_id: str):
    """删除报告"""
    try:
        success = ReportManager.delete_report(report_id)
        
        if not success:
            return jsonify({
                "success": False,
                "error": t('api.reportNotFound', id=report_id)
            }), 404
        
        return jsonify({
            "success": True,
            "message": t('api.reportDeleted', id=report_id)
        })
        
    except Exception as e:
        logger.error(f"删除报告失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


# ============== Report Agent对话接口 ==============

@report_bp.route('/chat', methods=['POST'])
def chat_with_report_agent():
    """
    与Report Agent对话
    
    Report Agent可以在对话中自主调用检索工具来回答问题
    
    请求（JSON）：
        {
            "simulation_id": "sim_xxxx",        // 必填，模拟ID
            "message": "请解释一下舆情走向",    // 必填，用户消息
            "chat_history": [                   // 可选，对话历史
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }
    
    返回：
        {
            "success": true,
            "data": {
                "response": "Agent回复...",
                "tool_calls": [调用的工具列表],
                "sources": [信息来源]
            }
        }
    """
    try:
        data = request.get_json() or {}
        
        simulation_id = data.get('simulation_id')
        message = data.get('message')
        chat_history = data.get('chat_history', [])
        
        if not simulation_id:
            return jsonify({
                "success": False,
                "error": t('api.requireSimulationId')
            }), 400

        if not message:
            return jsonify({
                "success": False,
                "error": t('api.requireMessage')
            }), 400
        
        # 获取模拟和项目信息
        manager = SimulationManager()
        state = manager.get_simulation(simulation_id)
        
        if not state:
            return jsonify({
                "success": False,
                "error": t('api.simulationNotFound', id=simulation_id)
            }), 404

        project = ProjectManager.get_project(state.project_id)
        if not project:
            return jsonify({
                "success": False,
                "error": t('api.projectNotFound', id=state.project_id)
            }), 404
        
        graph_id = state.graph_id or project.graph_id
        if not graph_id:
            return jsonify({
                "success": False,
                "error": t('api.missingGraphId')
            }), 400
        
        simulation_requirement = project.simulation_requirement or ""
        
        # 创建Agent并进行对话
        agent = ReportAgent(
            graph_id=graph_id,
            simulation_id=simulation_id,
            simulation_requirement=simulation_requirement
        )
        
        result = agent.chat(message=message, chat_history=chat_history)
        
        return jsonify({
            "success": True,
            "data": result
        })
        
    except Exception as e:
        logger.error(f"对话失败: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500
