
from sklearn.cluster import AgglomerativeClustering

import scipy.cluster


from scipy.cluster.hierarchy import dendrogram, linkage

import numpy as np

data= np.load(file="data_binary.npy")
leaf_labels=np.load(file="label_binary.npy")
print(np.shape(data))
X= np.array(data)
clusterer = AgglomerativeClustering(n_clusters=2,affinity='cosine', linkage='complete') # You can set compute_full_tree to 'auto', but I left it this way to get the entire tree plotted
clusterer.fit(X)
linked = linkage(X, 'ward')



T= scipy.cluster.hierarchy.to_tree(linked , rd=False )


import xml.etree.ElementTree as et

root = et.Element('root')
tree = et.ElementTree(element=root)
images = et.SubElement(root, 'Images')



def export_structure(T):

    node = et.SubElement(images, 'node')
    node_id = et.SubElement(node, 'id')
    node_id.text = str(T.id)
    node.attrib['is_leaf'] = 'no'

    if T.left:
        node_id = et.SubElement(node, 'leftChild')
        node_id.text = str(T.left.id)
        export_structure(T.left)

    if T.right:
        node_id = et.SubElement(node, 'rightChild')
        node_id.text = str(T.right.id)
        export_structure(T.right)

    else:
        node.attrib['is_leaf']='yes'
        node_id = et.SubElement(node, 'path')
        node_id.text = leaf_labels[T.id]



export_structure(T)

tree.write('images_Tree.xml', encoding='utf-8', xml_declaration=True)