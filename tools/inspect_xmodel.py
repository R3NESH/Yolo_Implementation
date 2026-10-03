"""Report what a compiled .xmodel will actually do on the board.

Run inside the Vitis AI container (it needs `xir`):

    cd yolov3_test && ./vitis_run.sh python ../tools/inspect_xmodel.py compiled_lrelu/*.xmodel

This answers the two questions that matter before copying anything to a board:

1. **How many DPU subgraphs?** More than one means the model is fragmented and
   `vart.Runner` will silently execute only the first of them. Use `GraphRunner`.
2. **Which ops fell back to the CPU?** Every CPU op needs a matching
   `/usr/lib/libvart_op_imp_<op>.so` on the board at runtime. If one is missing the model
   compiles fine and then aborts on the board - which is exactly what happens with
   `aten::silu_`. See the vault note 'SiLU Decomposition'.
"""

import sys
import collections

import xir


def describe(path):
    graph = xir.Graph.deserialize(path)
    root = graph.get_root_subgraph()
    children = root.toposort_child_subgraph()

    def device(sub):
        return sub.get_attr("device").upper() if sub.has_attr("device") else "?"

    by_device = collections.Counter(device(s) for s in children)
    dpu = [s for s in children if device(s) == "DPU"]

    print("=== %s" % path)
    print("  subgraphs: %s" % dict(by_device))

    cpu_ops = collections.Counter()
    for s in children:
        if device(s) == "CPU":
            for op in s.get_ops():
                cpu_ops[op.get_type()] += 1
    if cpu_ops:
        print("  CPU ops (each needs libvart_op_imp_<type>.so on the board):")
        for op_type, n in cpu_ops.most_common():
            print("    %-24s %d" % (op_type, n))
    else:
        print("  CPU ops: none")

    if dpu:
        first = dpu[0]
        print("  DPU[0] in : %s" % [(t.name, tuple(t.dims)) for t in first.get_input_tensors()])
        print("  DPU[0] out: %s" % [(t.name, tuple(t.dims)) for t in first.get_output_tensors()])

    if len(dpu) > 1:
        print("  !! %d DPU subgraphs. vart.Runner would execute only the first - use "
              "GraphRunner, and expect roughly one DPU<->CPU round trip per boundary." % len(dpu))
    elif len(dpu) == 1:
        print("  OK single DPU subgraph - vart.Runner is the fast path.")
    print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        describe(p)
