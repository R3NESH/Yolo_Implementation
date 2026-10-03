"""Print the DPU subgraph attributes the report's memory/compute tables are built from."""
import sys
import xir

g = xir.Graph.deserialize(sys.argv[1])
for sg in g.get_root_subgraph().toposort_child_subgraph():
    if sg.has_attr("device") and sg.get_attr("device") == "DPU":
        for k in ("workload", "workload_on_arch", "dpu_fingerprint"):
            print(k, sg.get_attr(k) if sg.has_attr(k) else None)
        for k in ("reg_id_to_size", "mc_code"):
            if sg.has_attr(k):
                v = sg.get_attr(k)
                print(k, v if k == "reg_id_to_size" else len(v))
        print("ops", len(sg.get_ops()))
