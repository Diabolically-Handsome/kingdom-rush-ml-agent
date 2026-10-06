import copy
import math
import unittest

from alpharush_rl.model_broker import prompt_hash, tokenize_request, validate_distribution, validate_request


class FakeTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return {"input_ids": [1, 17, 18]}

    def encode(self, label, add_special_tokens):
        # Prefix collision is intentional: AA must be scored as its complete
        # output sequence, never as A's first token alone.
        return {"A": [11], "AA": [11, 11], "B": [12]}[label]


class ModelBrokerContractTests(unittest.TestCase):
    def setUp(self):
        self.request = {"id": "native-1", "system": "Choose a label.", "user": "真实关卡",
                        "labels": ["A", "AA", "B"]}
        self.response = {"id": "native-1", "choice": "AA", "labels": self.request["labels"],
                         "p": [0.2, 0.5, 0.3], "logp": list(map(math.log, [0.2, 0.5, 0.3])),
                         "prompt_sha256": prompt_hash(self.request), "model": "instruction-base",
                         "complete_legal_distribution": True}

    def test_multitoken_labels_keep_eos_and_prefix_distinction(self):
        result = tokenize_request(FakeTokenizer(), self.request)
        self.assertEqual(result["suffixes"], [[11, 2], [11, 11, 2], [12, 2]])
        self.assertEqual(result["prompt_tokens"], 3)

    def test_complete_matched_distribution(self):
        self.assertIs(validate_distribution(self.response, self.request), self.response)

    def test_reject_top_n_or_different_prompt_or_label_order(self):
        for changed in ({"p": [0.2, 0.5]}, {"prompt_sha256": "0" * 64},
                        {"labels": ["AA", "A", "B"]}, {"id": "stale-native-0"}):
            response = copy.deepcopy(self.response)
            response.update(changed)
            with self.assertRaises(ValueError):
                validate_distribution(response, self.request)

    def test_request_cannot_hide_duplicate_or_unlisted_labels(self):
        request = copy.deepcopy(self.request)
        request["labels"] = ["A", "A"]
        with self.assertRaises(ValueError):
            validate_request(request)


if __name__ == "__main__":
    unittest.main()
