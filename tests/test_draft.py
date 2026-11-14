"""Unit tests for NGramDraftModel."""

import unittest
from thoughtformula.draft import NGramDraftModel


class TestNGramDraftModel(unittest.TestCase):
    def test_fit_and_draft(self):
        drafter = NGramDraftModel(max_n=4)
        seqs = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 6],
            [1, 2, 3, 4, 5],
        ]
        drafter.fit(seqs)

        drafted = drafter.draft([1, 2, 3], gamma=2)
        self.assertEqual(drafted, [4, 5])

    def test_draft_stop_token(self):
        drafter = NGramDraftModel(max_n=4)
        seqs = [[10, 20, 99]]
        drafter.fit(seqs)

        drafted = drafter.draft([10], gamma=3, stop_token_id=99)
        self.assertEqual(drafted, [20, 99])


if __name__ == "__main__":
    unittest.main()
