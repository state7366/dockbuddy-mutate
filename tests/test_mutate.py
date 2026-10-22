# -*- coding: utf-8 -*-
"""test_mutate.py —— 纯函数测试(不加载 OpenMM/PDBFixer)。"""
import unittest

from dockbuddy_mutate import (coordinate_span_A, list_engines, parse_mutation,
                              parse_mutations)


class TestParse(unittest.TestCase):
    def test_parse_mutation(self):
        m = parse_mutation("F205G")
        self.assertEqual((m.wt3, m.resnum, m.mut3), ("PHE", 205, "GLY"))

    def test_parse_mutations_multi(self):
        muts = parse_mutations(["F205G,S126A"], default_chain="A")
        self.assertEqual(len(muts), 2)
        self.assertEqual(muts[0].resnum, 205)
        self.assertEqual(muts[1].resnum, 126)

    def test_list_engines(self):
        engines = list_engines()
        self.assertTrue(any(e["name"] == "pdbfixer" for e in engines))


if __name__ == "__main__":
    unittest.main(verbosity=2)
