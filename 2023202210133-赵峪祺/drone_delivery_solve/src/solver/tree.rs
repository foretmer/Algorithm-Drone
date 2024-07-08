use std::collections::HashMap;

pub struct Node {           // save the son and parent idx
    pub data: usize,

    pub index: usize,
    pub parent: usize,
    pub sons: Vec<usize>,
}

pub struct Tree {
    pub nodes: Vec<Node>,
    pub root: usize,

    pub d_to_n: HashMap<usize, usize>,
}

impl Tree {
    pub fn new_with_root(root_data: usize) -> Self {
        Self { 
            nodes: vec![Node {
                data: root_data,
                sons: Vec::new(),
                parent: 0,
                index: 0,
            }], 
            root: 0,
            d_to_n: vec![(root_data, 0)]
                .into_iter()
                .collect::<HashMap<_, _>>(),
        }
    }

    pub fn get_node(&self, index: &usize) -> &Node {
        &self.nodes[*index]
    }

    pub fn get_path(&self) -> Vec<usize> {
        let mut whole_path: Vec<usize> = Vec::with_capacity(self.nodes.len());
        let now_node = self.get_node(&self.root);

        self.pre_travel(now_node, &mut whole_path);
        whole_path
    }

    fn pre_travel(&self, node: &Node, path: &mut Vec<usize>) {
        path.push(node.data);

        node.sons.iter().for_each(|son| {
            self.pre_travel(self.get_node(son), path);
        })
    }

    pub fn data_exist(&self, data: &usize) -> bool {
        self.d_to_n.contains_key(data)
    }

    pub fn insert_node_by_index(&mut self, par_index: usize, data: usize) {
        let last_index = self.nodes.len().to_owned();

        let new_node = Node {
            data: data,
            index: last_index,
            parent: par_index,
            sons: Vec::new(),
        };

        self.nodes[par_index].sons.push(last_index);
        self.d_to_n.insert(data, last_index);
        self.nodes.push(new_node);
    }

    pub fn insert_node_by_data(&mut self, data_par: &usize, data_son: usize) {
        self.insert_node_by_index(self.d_to_n[data_par], data_son);
    }
}

