import json
import uuid
from unittest.mock import patch

from django.test import TestCase

from .models import ChatMessage, ChatSession


class SessionEndpointsTests(TestCase):
    def setUp(self):
        self.client_id = uuid.uuid4()
        self.other_client_id = uuid.uuid4()
        self.base_url = "/api/chat/sessions/"

    def test_list_sessions_requires_client_id(self):
        resp = self.client.get(self.base_url)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.json())

    def test_list_sessions_returns_only_requested_client(self):
        s1 = ChatSession.objects.create(client_id=self.client_id, title="A")
        ChatSession.objects.create(client_id=self.other_client_id, title="B")

        resp = self.client.get(self.base_url, {"client_id": str(self.client_id)})
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(len(payload["sessions"]), 1)
        self.assertEqual(payload["sessions"][0]["id"], str(s1.id))

    def test_session_detail_get_and_delete(self):
        session = ChatSession.objects.create(client_id=self.client_id, title="Chat")
        ChatMessage.objects.create(session=session, role="user", content="hi")
        ChatMessage.objects.create(session=session, role="assistant", content="hello")

        detail_url = f"/api/chat/sessions/{session.id}/"
        get_resp = self.client.get(detail_url, {"client_id": str(self.client_id)})
        self.assertEqual(get_resp.status_code, 200)
        self.assertEqual(len(get_resp.json()["messages"]), 2)

        del_resp = self.client.delete(detail_url, {"client_id": str(self.client_id)})
        self.assertEqual(del_resp.status_code, 200)
        self.assertFalse(ChatSession.objects.filter(id=session.id).exists())


class ChatCompletionEndpointTests(TestCase):
    def setUp(self):
        self.url = "/api/chat/"

    def test_chat_completion_invalid_json(self):
        resp = self.client.post(self.url, data="{bad", content_type="application/json")
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"], "Invalid JSON")

    def test_chat_completion_requires_message_or_messages(self):
        resp = self.client.post(
            self.url,
            data=json.dumps({}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.json())

    @patch("chat.chat_logic.call_llm", return_value=("Mock reply", None))
    @patch(
        "chat.chat_logic.prepare_chat_prompts",
        return_value=("system", "user prompt", {"sources": []}),
    )
    def test_chat_completion_stateless_success(self, _mock_prepare, _mock_call):
        resp = self.client.post(
            self.url,
            data=json.dumps({"message": "hello"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["reply"], "Mock reply")
        self.assertIn("rag", payload)

    @patch("chat.chat_logic.call_llm", return_value=("DB reply", None))
    @patch(
        "chat.chat_logic.prepare_chat_prompts",
        return_value=("system", "user prompt", {"sources": []}),
    )
    def test_chat_completion_session_persists_messages(self, _mock_prepare, _mock_call):
        cid = str(uuid.uuid4())
        resp = self.client.post(
            self.url,
            data=json.dumps({"client_id": cid, "message": "hello db"}),
            content_type="application/json",
        )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertIn("session_id", payload)
        session = ChatSession.objects.get(id=payload["session_id"])
        self.assertEqual(str(session.client_id), cid)
        self.assertEqual(session.messages.count(), 2)
