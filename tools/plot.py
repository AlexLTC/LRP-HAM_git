import networkx as nx
import matplotlib.pyplot as plt
# colors = ['red', 'green', 'blue', 'yellow','black']
colors = [127/255,50/255,200/255]

DG = nx.DiGraph()  #一次性新增多節點，輸入的格式為列表
# DG.add_nodes_from(['lip', 'teeth', 'tongue', 'Pharynx', 'Epiglottis', 'Arytenoid cartilage', 'vocal cord', 'Throat'])
# DG.add_edges_from([('lip', 'teeth'), ('teeth', 'tongue'), ('tongue', 'lip'), ('tongue', 'Pharynx'), ('Pharynx', 'Epiglottis'), \
#                    ('Epiglottis', 'Arytenoid cartilage'), ('Arytenoid cartilage', 'vocal cord'), ('vocal cord', 'Epiglottis'), \
#                    ('vocal cord', 'Throat'),\
#                    ('Pharynx', 'Epiglottis'), ('Pharynx', 'Epiglottis')])
DG.add_nodes_from(['Epiglottis', 'Arytenoid cartilage', 'vocal cord'])
DG.add_edges_from([('Epiglottis', 'Arytenoid cartilage'), ('Arytenoid cartilage', 'vocal cord'), ('vocal cord', 'Epiglottis')])
nx.draw(DG,with_labels=True, node_size=1500, node_color = colors)
plt.show()