# C2-GMAE Code Release

This repository contains the node-level and graph-level experimental code.

## Structure

```text
node_level/      Node classification experiments.
graph_level/     Graph property prediction experiments.
```

## Node-level example

```bash
cd node_level
cp config/blog.yaml config/blog.yaml
python train_node.py --dataset blog --root ./dataset --config_dir ./config
```

The command above loads `node_level/dataset/blog.pt` and `node_level/config/blog.yaml`.

## Graph-level example

```bash
cd graph_level
cp config/ogbg-molbace.yaml config/ogbg-molbace.yaml
python train_graph.py --dataset ogbg-molbace --root ./dataset --config_dir ./config
```
