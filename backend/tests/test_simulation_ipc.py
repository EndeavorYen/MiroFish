"""File-based IPC between the backend and a running simulation (#68)."""

import json
import os
import threading
import time

import pytest

from app.services.simulation_ipc import (
    CommandStatus,
    CommandType,
    SimulationIPCClient,
    SimulationIPCServer,
)


def _serve_once(server: SimulationIPCServer, reply) -> threading.Thread:
    def run():
        for _ in range(200):
            command = server.poll_commands()
            if command is not None:
                reply(command)
                return
            time.sleep(0.01)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_a_command_round_trips_and_its_files_are_removed(tmp_path):
    client = SimulationIPCClient(str(tmp_path))
    server = SimulationIPCServer(str(tmp_path))
    seen = []

    def reply(command):
        seen.append(command)
        server.send_success(command.command_id, {"echo": command.args["prompt"]})

    thread = _serve_once(server, reply)
    response = client.send_command(CommandType.INTERVIEW, {"agent_id": 3, "prompt": "你怎麼看？"},
                                   timeout=5, poll_interval=0.01)
    thread.join(timeout=5)
    assert response.status == CommandStatus.COMPLETED
    assert response.result == {"echo": "你怎麼看？"}
    assert seen[0].command_type == CommandType.INTERVIEW and seen[0].args["agent_id"] == 3
    assert os.listdir(tmp_path / "ipc_commands") == [] and os.listdir(tmp_path / "ipc_responses") == []


def test_an_error_reply_reaches_the_client(tmp_path):
    client = SimulationIPCClient(str(tmp_path))
    server = SimulationIPCServer(str(tmp_path))
    thread = _serve_once(server, lambda c: server.send_error(c.command_id, "agent 9 does not exist"))
    response = client.send_command(CommandType.INTERVIEW, {"agent_id": 9}, timeout=5, poll_interval=0.01)
    thread.join(timeout=5)
    assert response.status == CommandStatus.FAILED and response.error == "agent 9 does not exist"


def test_no_reply_times_out_and_withdraws_the_command(tmp_path):
    client = SimulationIPCClient(str(tmp_path))
    with pytest.raises(TimeoutError):
        client.send_command(CommandType.CLOSE_ENV, {}, timeout=0.1, poll_interval=0.02)
    # Withdrawn: a server that has not polled yet will not act on it (one
    # that already picked it up still writes a response nobody reads).
    assert os.listdir(tmp_path / "ipc_commands") == []


def test_the_server_takes_the_oldest_command_and_skips_broken_files(tmp_path):
    server = SimulationIPCServer(str(tmp_path))
    commands = tmp_path / "ipc_commands"
    (commands / "broken.json").write_text("{not json", encoding="utf-8")
    for index, command_id in enumerate(("first", "second")):
        path = commands / f"{command_id}.json"
        path.write_text(json.dumps({"command_id": command_id, "command_type": "interview", "args": {}}),
                        encoding="utf-8")
        os.utime(path, (1000 + index, 1000 + index))
    os.utime(commands / "broken.json", (500, 500))
    assert server.poll_commands().command_id == "first"


def test_env_status_follows_the_server(tmp_path):
    client = SimulationIPCClient(str(tmp_path))
    server = SimulationIPCServer(str(tmp_path))
    assert client.check_env_alive() is False  # no status file yet
    server.start()
    assert client.check_env_alive() is True
    server.stop()
    assert client.check_env_alive() is False
    (tmp_path / "env_status.json").write_text("{half", encoding="utf-8")
    assert client.check_env_alive() is False  # a torn write reads as not alive


def test_the_client_and_the_simulation_process_handler_agree(tmp_path):
    """The handler the simulation process actually runs (scripts/sim_ipc.py)."""

    from scripts.run_parallel_simulation import ParallelIPCHandler

    client = SimulationIPCClient(str(tmp_path))
    handler = ParallelIPCHandler(str(tmp_path))

    def run():
        for _ in range(200):
            command = handler.poll_command()
            if command is not None:
                handler.send_response(command["command_id"], "completed", result={"ok": command["args"]["n"]})
                return
            time.sleep(0.01)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    response = client.send_command(CommandType.BATCH_INTERVIEW, {"n": 2}, timeout=5, poll_interval=0.01)
    thread.join(timeout=5)
    assert response.status == CommandStatus.COMPLETED and response.result == {"ok": 2}
    assert os.listdir(tmp_path / "ipc_commands") == [] and os.listdir(tmp_path / "ipc_responses") == []
