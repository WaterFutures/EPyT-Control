import numpy as np


def get_edge_attribute(topology, name):
    return np.array(
        [ topology.get_link_info(l)[name] for l, _ in topology.get_all_links() ]
    )

def make_edge_index(topo, nodelist=None, bidirectional=False):
    edges, edge_index = zip(*map(
        lambda l: (l[0], get_node_index(topo, l[1])),
        filter(
            lambda l: l[1][0] in nodelist and l[1][1] in nodelist, 
            topo.get_all_links()
        )
        if nodelist is not None else 
        topo.get_all_links()
    ))
    edge_index = np.stack(edge_index, 1)
    if bidirectional:
        return np.concatenate((edge_index, edge_index[::-1]), axis=1)
    return edge_index

def flow_to_velocity(topology, flow_data, unit='CMH'):
    assert unit == 'CMH'
    diameters = get_edge_attribute(topology, 'diameter')
    diameters = diameters / 10 / 100 # convert mm to meters
    diameters = np.broadcast_to(diameters[None], flow_data.shape)
    crosssection = (diameters / 2)**2 * np.pi
    crosssection[crosssection == 0] = 1e-10 # TODO: Changed to 1e-10 flow velocity through 0-diameter pipes are infinite
    flow_velocities = flow_data / crosssection
    flow_velocities = np.nan_to_num(flow_velocities) # [m/h]
    return flow_velocities / 60 / 60 # [m/s]

def get_node_index(topology, nodes):
    if isinstance(nodes, str):
        nodes = [nodes]
    return [ topology.get_all_nodes().index(n) for n in nodes ]
