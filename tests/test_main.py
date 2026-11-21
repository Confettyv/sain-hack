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


class Title(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), main.Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        main.State.cache.clear()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def titled(self, text: str, content: str, fake: bool = False) -> str:
        chat = mock.Mock()
        chat.ask.return_value = main.llm.Reply(content=content, tokens_in=0, out=0, finish="stop")
        body = json.dumps({"text": text, "fake": fake}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.server.server_port}/title", body, {"Content-Type": "application/json"})
        with mock.patch.object(main.llm, "from_env", return_value=chat), urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())["title"]

    def test_model_title_cleaned(self) -> None:
        self.assertEqual(self.titled("Пью кофе в пять вечера", '{"title": "«Кофе вечером и сон»."}'), "Кофе вечером и сон")

    def test_broken_reply_gives_empty(self) -> None:
        self.assertEqual(self.titled("павп", "не json"), "")

    def test_fake_mode_skips_model(self) -> None:
        self.assertEqual(self.titled("Пью кофе", '{"title": "Кофе"}', fake=True), "")


class Preview(unittest.TestCase):
    def test_og_image_served(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), main.Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/og.jpg", timeout=5) as resp:
                self.assertEqual(resp.headers["Content-Type"], "image/jpeg")
                self.assertGreater(len(resp.read()), 1000)
        finally:
            server.shutdown()
            server.server_close()


class DatasetCode(unittest.TestCase):
    def tearDown(self) -> None:
        main.State.dataset = None

    def loaded_codes(self, env: dict[str, str], fails: tuple[str, ...] = ()) -> list[str]:
        main.State.dataset = None

        def answer(case: int, code: str, auth: str) -> dict:
            if code in fails:
                raise main.DataError("HTTP 403")
            return {"dataset": dataset()}

        with mock.patch.dict(main.os.environ, env, clear=True), mock.patch.object(main, "fetch", side_effect=answer) as f:
            main.dataset()
        return [c.args[1] for c in f.call_args_list]

    def test_code_from_env(self) -> None:
        self.assertEqual(self.loaded_codes({"HACK_TEAM_TOKEN": "t", "DATASET_CODE": "case4-x"}), ["case4-x"])

    def test_final_by_default(self) -> None:
        self.assertEqual(self.loaded_codes({"HACK_TEAM_TOKEN": "t"}), [main.FINAL_CODE])

    def test_dev_when_final_closed(self) -> None:
        self.assertEqual(self.loaded_codes({"HACK_TEAM_TOKEN": "t"}, (main.FINAL_CODE,)), [main.FINAL_CODE, main.DEV_CODE])


if __name__ == "__main__":
    unittest.main()
