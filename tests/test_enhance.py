import unittest
from unittest.mock import Mock, patch

from ai import enhance


class EnhanceTests(unittest.TestCase):
    @patch("ai.enhance.ChatOpenAI")
    def test_qwen_thinking_is_disabled_for_structured_output(self, chat_openai):
        chat_openai.return_value.with_structured_output.return_value = Mock()

        enhance.create_structured_llm("qwen3.7-plus")

        chat_openai.assert_called_once_with(
            model="qwen3.7-plus",
            extra_body={"enable_thinking": False},
        )

    @patch("ai.enhance.ChatOpenAI")
    def test_other_models_do_not_receive_qwen_parameter(self, chat_openai):
        chat_openai.return_value.with_structured_output.return_value = Mock()

        enhance.create_structured_llm("deepseek-chat")

        chat_openai.assert_called_once_with(model="deepseek-chat")

    @patch("ai.enhance.create_chain")
    def test_preflight_stops_after_first_provider_error(self, create_chain):
        chain = Mock()
        chain.invoke.side_effect = RuntimeError("provider rejected request")
        create_chain.return_value = chain
        data = [
            {"id": "paper-1", "summary": "first"},
            {"id": "paper-2", "summary": "second"},
        ]

        with self.assertRaisesRegex(RuntimeError, "AI preflight failed"):
            enhance.process_all_items(data, "qwen3.7-plus", "Chinese", 1)

        chain.invoke.assert_called_once()


if __name__ == "__main__":
    unittest.main()
