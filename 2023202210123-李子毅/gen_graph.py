import networkx as nx
import matplotlib.pyplot as plt
import csv

from const import NUM_DC


# 读取CSV文件并创建图
def create_graph_from_csv(filename):
    G = nx.Graph()
    with open(filename, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过标题行
        for row in reader:
            from_id, to_id, distance = row
            distance = float(distance)
            # 检查是否为配送中心或卸货点，并添加相应的节点类型
            if int(from_id) <= NUM_DC:  # 配送中心编号为1-5
                from_type = "dc"
            else:  # 卸货点编号为6-10
                from_type = "dp"
            if int(to_id) <= NUM_DC:
                to_type = "dc"
            else:
                to_type = "dp"

            # 确保配送中心和卸货点作为节点被添加到图中
            G.add_node(from_id, type=from_type)
            G.add_node(to_id, type=to_type)
            # 添加边，并包含权重属性
            G.add_edge(from_id, to_id, weight=distance)
    return G


def set_node_properties(G):
    dc_color = "green"
    dp_color = "blue"
    for node, attr in G.nodes(data=True):
        if attr["type"] == "dc":
            attr["color"] = dc_color
        elif attr["type"] == "dp":
            attr["color"] = dp_color


if __name__ == "__main__":
    # 创建图
    G = create_graph_from_csv("drone_delivery_data.csv")
    set_node_properties(G)
    # 绘制图
    pos = nx.spring_layout(G)  # 为图形设置布局
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=list(nx.get_node_attributes(G, "color").values()),  # 使用设置的颜色
        edge_color="#FF5733",
        node_size=700,
        font_size=12,
        font_weight="bold",
        linewidths=1,
        font_color="white",
        alpha=0.6,
    )

    # 保存图
    plt.savefig("drone_delivery_network.png")
    # 显示图
    plt.show()
