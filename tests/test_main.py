import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

import main
from sain.data import Dataset
from tests.test_pipeline import dataset

REPLY = {"refused": False, "citations": [], "answers": [{"expert_id": "clinician", "text": "ok"}]}


class AskUser(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), main.Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        main.State.cache.clear()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        main.State.dataset = None

    def asked_user(self, users: list[dict]) -> str:
        main.State.dataset = Dataset(dataset() | {"users": users})
        body = json.dumps({"expert_id": "clinician", "text": "Почему я просыпаюсь в четыре?", "user_id": "u2", "fake": True})
        url = f"http://127.0.0.1:{self.server.server_port}/ask"
        with mock.patch.object(main, "ask_live", return_value=dict(REPLY)) as live:
            req = urllib.request.Request(url, body.encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
        return live.call_args.args[4]

    def test_single_user_of_last_run(self) -> None:
        self.assertEqual(self.asked_user([{"user_id": "u1"}]), "u1")

    def test_several_users_stay_anonymous(self) -> None:
        self.assertEqual(self.asked_user([{"user_id": "u1"}, {"user_id": "u2"}]), "")


class DatasetCode(unittest.TestCase):
    def tearDown(self) -> None:
        main.State.dataset = None

    def loaded_code(self, env: dict[str, str]) -> str:
        main.State.dataset = None
        with mock.patch.dict(main.os.environ, env), mock.patch.object(main, "fetch", return_value={"dataset": dataset()}) as f:
            main.dataset()
        return f.call_args.args[1]

    def test_final_code_from_env(self) -> None:
        self.assertEqual(self.loaded_code({"HACK_TEAM_TOKEN": "t", "DATASET_CODE": "case4-final"}), "case4-final")

    def test_dev_by_default(self) -> None:
        with mock.patch.dict(main.os.environ, {}, clear=False):
            main.os.environ.pop("DATASET_CODE", None)
            self.assertEqual(self.loaded_code({"HACK_TEAM_TOKEN": "t"}), main.DEV_CODE)


if __name__ == "__main__":
    unittest.main()
