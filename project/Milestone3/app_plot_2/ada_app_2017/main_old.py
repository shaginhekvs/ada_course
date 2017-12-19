from bokeh.layouts import widgetbox, column, row
from bokeh.models import ColumnDataSource
from bokeh.models.markers import Circle
from bokeh.models.glyphs import Text
from bokeh.models.widgets import Panel,Tabs,Div,Paragraph
from bokeh.plotting import figure, curdoc
from bokeh.models import HoverTool, Select, Button , Slider
import community # package name: python-louvain
from math import sqrt
import networkx 
import pandas as pd
import numpy as np
import pickle
from networkx.algorithms import centrality as centrality_algorithms
from os.path import dirname, join
from bokeh.palettes import Blues4,Reds4,Greens4

color_for = Blues4
color_anti = Reds4

def plot_formatter(p,title='Variation of number of tweets over time',y_text='Count of tweets',x_text='Year'):
    p.legend.location='top_left'
    p.legend.click_policy="mute"
    p.legend.label_text_font='helvetica'
    p.legend.label_text_font_size='12pt'
    p.xgrid.visible=False
    p.background_fill_color = "beige"
    p.background_fill_alpha = 0.2
    p.axis.major_label_text_font='helvetica'
    p.axis.major_label_text_font_size='12pt'
    p.axis.major_label_text_color='gray'
    p.axis.axis_line_color='gray'
    p.axis.axis_label_text_font_size='14pt'
    p.axis.axis_label_text_font='helvetica'
    p.axis.axis_label_text_color='gray'
    p.xaxis.axis_label = 'Year'
    p.yaxis.axis_label=y_text
    p.legend.click_policy="mute"
    p.title.text=title
    p.title.text_font_size='14pt'
    p.title.text_font='helvetica'
    return p

class Create_time_count_plot():
     
    def __init__(self,source_df,text_for,source_anti_df=None,text_anti=None):
        
        source_df=self.read_file(source_df)
        source_df['legend']=text_for
        source_df.dropna(inplace=True)
        self.source_for = ColumnDataSource(dict(x=source_df.index, y=source_df['count'],
                                     hashes=source_df.hashes_inside,date=source_df.date,\
                                     legend=source_df.legend,text=source_df.text))
        
        self.text_for = text_for
        if(source_anti_df == None):
            self.source_anti = source_anti_df
        else:
            source_anti_df=self.read_file(source_anti_df)
            source_anti_df.dropna(inplace=True)
            source_anti_df['legend']=text_anti
            self.source_anti = ColumnDataSource(dict(x=source_anti_df.index, y=source_anti_df['count'],
                                     hashes=source_anti_df.hashes_inside,date=source_anti_df.date,\
                                     legend=source_anti_df.legend,text=source_anti_df.text))
        self.text_anti = text_anti
    
        self.plot_n_lines = self.create_plot(self.source_for,self.text_for,source_anti=self.source_anti,text_anti=self.text_anti)
    
    
    def read_file(self,path):
        df=pickle.load(open(path,'rb'))
        return df
    
    def reinit(self,source_df,text_for,source_anti_df=None,text_anti=None):
        source_df=self.read_file(source_df)
        source_df.dropna(inplace=True)
        source_df['legend']=text_for
        self.source_for.data['x']=source_df.index
        self.source_for.data['y']=list(source_df['count'].values)
        self.source_for.data['hashes']=source_df.hashes_inside
        self.source_for.data['text']=source_df.text
        self.source_for.data['date']=source_df.date
        self.source_for.data['legend']=source_df['legend']
        if(source_anti_df!=None and self.source_anti!=None):
            source_anti_df=self.read_file(source_anti_df)
            source_anti_df.dropna(inplace=True)
            source_anti_df['legend']=text_anti
            self.source_anti.data['x']=source_anti_df.index
            self.source_anti.data['y']=source_anti_df['count']
            self.source_anti.data['hashes']=source_anti_df.hashes_inside
            self.source_anti.data['date']=source_anti_df.date   
            self.source_anti.data['legend']=source_anti_df['legend']
            self.source_anti.data['text']=source_anti_df.text
        
    def create_plot(self,source_for,text_for,source_anti=None,text_anti=None):
        p = figure(plot_width=1600, plot_height=400,tools=['tap','box_zoom', 'reset','pan','wheel_zoom'],x_axis_type="datetime")
        l1 = p.line('x','y', line_width=2, color=color_for[0],source=source_for,muted_alpha=0.2,legend='legend')
        l1_hover = HoverTool(renderers=[l1],
                             tooltips=[('Date', '@date'), ('count', '@y'),('Top Hashes','@hashes'),('Top words','@text')],mode='vline')
        
        cr1 = p.circle('x', 'y', size=10,
                    fill_color="grey", hover_fill_color=color_for[1],
                    fill_alpha=0.05, hover_alpha=0.3,
                    line_color=None, hover_line_color="white",source=source_for)
        
        p.min_border_left = 100
        p.add_tools(l1_hover)
        
        if(source_anti==None):
            p.add_tools(HoverTool(tooltips=None, renderers=[cr1], mode='vline')),[l1_hover]
            
            return plot_formatter(p),[l1]
        l2 = p.line('x','y', line_width=2, color=color_anti[0],source=source_anti,muted_alpha=0.2,legend='legend')
        l2_hover = HoverTool(renderers=[l2],
                             tooltips=[('Date', '@date'), ('count', '@y'),('Top Hashes','@hashes'),('Top words','@text')],mode='vline')
    
        p.add_tools(l2_hover)
        
        cr = p.circle('x', 'y', size=10,
                    fill_color="grey", hover_fill_color=color_anti[1],
                    fill_alpha=0.05, hover_alpha=0.3,
                    line_color=None, hover_line_color="white",source=source_anti)
        
        p.add_tools(HoverTool(tooltips=None, renderers=[cr,cr1], mode='vline'))
    
        return plot_formatter(p),[l1,l2]
    
    def return_view(self):
        return row(column(widgetbox(Div(text='<h1> \n \t   </h1>',width=200)),self.plot_n_lines[0]))
    

class Create_polarity_plot():
    
    def __init__(self,source_for,label_for,source_anti=None,label_anti='',title='Polarity',y_axis='Sentiments'):
        self.path_for=source_for
        self.path_anti=source_anti
        self.source_for=self.load_data(self.path_for)
        self.source_for['legend']=label_for
        self.source_for=ColumnDataSource(data=self.source_for)
        if(source_anti!=None):
            self.source_anti=self.load_data(self.path_anti)
            self.source_anti['legend']=label_anti
            self.source_anti=ColumnDataSource(data=self.source_anti)
        else:
            self.source_anti=None
            
        self.plot=self.make_plot(self.source_for , self.source_anti,title,y_axis)
        
    def return_view(self):
        return row(self.plot)
    
    def reinit(self,source_for,label_for,source_anti,label_anti,title='Polarity'):
        self.plot.title.text=title
        self.path_for=source_for
        self.path_anti=source_anti
        source_df=self.load_data(self.path_for)
        source_df['legend']=label_for
        for key in self.source_for.data.keys():
            self.source_for.data[key]=source_df[key].values
        
        if(self.source_anti!=None):
            source_df=self.load_data(self.path_anti)
            source_df['legend']=label_anti
            for key in self.source_anti.data.keys():
                self.source_anti.data[key]=source_df[key].values
            
        
         
        
    def make_plot(self,source, source2,title,y_label):
        plot = figure(x_axis_type="datetime", plot_width=1600,plot_height=400,tools=['tap','box_zoom', 'reset','pan','wheel_zoom'])
        plot.title.text = title
    
        q1=plot.quad(top='max', bottom='min', left='left', right='right',
                  color=color_for[2], source=source,alpha=0.3, hover_fill_color=Reds4[0],\
                    line_color=None, hover_line_color="white",hover_alpha=0.6)
        
        l1=plot.line(x='date',y='mean',line_width=2,line_color=color_for[0],source=source,legend="legend",muted_alpha=0.2,alpha=0.8)
        
        tooltips="""
        <div>
            <div>
                        <span style="font-size: 12px;">Date is @date_str ,mean is @mean{.00}, std is @std{.00}</span>
            </div>
        </div>
        """      
        
        if(source2==None):
            plot.add_tools(HoverTool(tooltips=tooltips, renderers=[q1], mode='vline'))
            return plot_formatter(plot , title = title , y_text=y_label)
            
        q2=plot.quad(top='max', bottom='min', left='left', right='right',
                  color=color_anti[2], source=source2,alpha=0.2, hover_fill_color=Blues4[0],\
                    line_color=None, hover_line_color="white",hover_alpha=0.6)
        
        l2=plot.line(x='date',y='mean',line_width=2,line_color=color_anti[0],source=source2,legend="legend",muted_alpha=0.2,alpha=0.8)
        

        hover = HoverTool(tooltips=tooltips, renderers=[q1,q2], mode='vline')
        plot.add_tools(hover)
        return plot_formatter(plot , title = title , y_text=y_label)
    
    def load_data(self,path):
        return pickle.load(open(path,'rb'))
    
    
        
class Create_percent_sentiment_plot():
    
    def __init__(self,source_path,kind,title_name,events_path=None,min_val=0,max_val=1,width=800):
        title=kind + ' of ' +title_name
        data=self.load_files(source_path,kind,events_path,min_val,max_val)
        self.source=ColumnDataSource(data=data[0])
        if(kind=='Polarity'):
            labels = ['Negative','Neutral','Positive']
        else:
            labels = ['Subjective' , 'Neutral' , 'Rational' ]          

        if(len(data)>1):
            self.events=ColumnDataSource(data=data[1])
        else:
            self.events=None
            
        self.plot=self.create_percent_plot(self.source,title,labels,width,self.events)

    def load_files(self,source_path,kind,events_path,min_val,max_val):
        source=pickle.load(open(source_path,'rb'))
        if(events_path):
            events=pickle.load(open(events_path,'rb'))
            events['y1']=max_val
            events['y0']=min_val
            return [source,events]
        else:
            return [source]
        
    def create_percent_plot(self,source,title,labels,width,events=None):
        p = figure(plot_width=width, plot_height=400,tools=['tap','box_zoom', 'reset','pan','wheel_zoom'],x_axis_type="datetime")
        l1 = p.line('date','Neg', line_width=2, color='#ffeb6b',source=source,muted_alpha=0.2,legend=labels[0])
        l1_hover = HoverTool(renderers=[l1],
                             tooltips=[('Date', '@date_str'), ('fraction', '@Neg{0.00}')],mode='vline')
        
        cr1 = p.circle('date', 'Neg', size=10,
                    fill_color="grey", hover_fill_color='#ffeb6b',
                    fill_alpha=0.05, hover_alpha=0.3,
                    line_color=None, hover_line_color="white",source=source)
        
        
        p.add_tools(l1_hover)
    
        l2 = p.line('date','0', line_width=2, color=Blues4[0],source=source,muted_alpha=0.2,legend=labels[1])
        l2_hover = HoverTool(renderers=[l2],
                             tooltips=[('Date', '@date_str'), ('fraction', '@0{0.00}')],mode='vline')
    
        p.add_tools(l2_hover)
        
        cr2 = p.circle('date', '0', size=10,
                    fill_color="grey", hover_fill_color=Blues4[0],
                    fill_alpha=0.05, hover_alpha=0.3,
                    line_color=None, hover_line_color="white",source=source)
        
        p.add_tools(HoverTool(tooltips=None, renderers=[cr2,cr1], mode='vline'))
        
        l3 = p.line('date','Pos', line_width=2, color=Greens4[0],source=source,muted_alpha=0.2,legend=labels[2])
        l3_hover = HoverTool(renderers=[l3],
                             tooltips=[('Date', '@date_str'), ('fraction', '@Pos{0.00}')],mode='vline')
        
        cr3 = p.circle('date', 'Pos', size=10,
                    fill_color="grey", hover_fill_color=Greens4[0],
                    fill_alpha=0.05, hover_alpha=0.3,
                    line_color=None, hover_line_color="white",source=source)
        
    
        p.add_tools(l3_hover)
        
        p.add_tools(HoverTool(tooltips=None, renderers=[cr2,cr1,cr3], mode='vline'))
        
        if(events!=None):
            vb=p.vbar(x='date',width='width',bottom='y0',top='y1',source=events,color='gray',alpha=0.4)
        
            vb2=p.vbar(x='date',width='width_inv',bottom='y1',top='y1',source=events,color='white',alpha=0.1)
        
            vb_hover = HoverTool(renderers=[vb2],
                             tooltips=[('Date', '@date_str'), ('News', '@event')],mode='vline')
        
            p.add_tools(vb_hover)
        return plot_formatter(p,title=title,y_text='Fraction',x_text='Year')
    
    def return_view(self):
        return self.plot
    
    def reinit(self,source_path,kind,title_name,events_path,min_val=0,max_val=1):
        data=self.load_files(source_path,kind,events_path,min_val,max_val)
        title=kind + ' of ' +title_name
        self.plot.title.text=title
        for key in self.source.data.keys():
            if key in data[0].columns:
                self.source.data[key]=data[0][key].values
        if(len(data)>1 and self.events!=None):
            
            for key in self.events.data.keys():
                if key in data[1].columns:
                    self.events.data[key]=data[1][key].values
        
        
            


class Create_image_html():
    
    def __init__(self,image_path,text):
        self.space=Paragraph(text='')
        self.para=Div(text='<h2 style="color:#3b3a39;text-align:center;font-size:150%;font-family:Times New Roman;">{}</h1>'.format(text),width=900)
        
        self.div=Div(text='<img src="{}" alt="{}" border="0" height="500" width="800"></img>'.format(image_path,text),width=900,height=500)
        self.div.css_classes = ["custom"]
    def return_view(self):
        return column(self.space,self.para,self.div)
    
    def reinit(self,image_path,text):
        self.para.text='<h2 style="color:#3b3a39;text-align:center;font-size:150%;font-family:Times New Roman;">{}</h1>'.format(text)
        self.div.text='<img src="{}" alt="{}" border="0" height="500" width="800"></img>'.format(image_path,text)
    
    
class Create_LDA_html():
    
    def __init__(self,lda_path,title):
        self.space=Paragraph(text='')
        self.para=Div(text='<h2 style="color:#3b3a39;text-align:center;font-size:150%;font-family:Times New Roman;">{}</h1>'.format(title),width=900)
        self.div=Div(text='{}'.format(self.read_file(lda_path)),width=900,height=500)

    def return_view(self):
        return column(self.space,self.para,self.div)
    
    def read_file(self,path):
        return open(path).read()
    
    def reinit(self,lda_path,text):
        self.para.text='<h2 style="color:#3b3a39;text-align:center;font-size:150%;font-family:Times New Roman;">{}</h1>'.format(text)
        self.div.text='{}'.format(self.read_file(lda_path))
    
    

class Create_network():
    
    
    centrality_metrics = {"Degree Centrality":
                          lambda n, weight='_': centrality_algorithms.degree_centrality(n),
                      "Closeness Centrality":
                          lambda n, weight='_': centrality_algorithms.closeness_centrality(n),
                      "Betweenness Centrality":
                          centrality_algorithms.betweenness_centrality}

    community_colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628', \
                            '#b3cde3','#ccebc5','#decbe4','#fed9a6','#ffffcc','#e5d8bd','#fddaec',\
                            '#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']

    def __init__(self,network_file,layout_file,count_path,title,width=800,thresh_val=8):
        self.network_file=network_file
        self.layout_file=layout_file
        self.count_path=count_path
        self.network_tuple = self.load_network(network_file,layout_file)
        self.nodes_sources_tab1 = self.column_source(self.network_tuple[1],count_path)
        self.network_plots_n_circle_tab1 = self.create_network_plot(self.nodes_sources_tab1,title,width)
        self.network_lines_tab1 = self.add_lines(self.network_tuple,self.network_plots_n_circle_tab1[0])
        self.get_centrality_n_community(self.network_tuple[0],self.nodes_sources_tab1,self.network_plots_n_circle_tab1[1])
        self.drop_button_tab1 = Button(label="Remove Node", button_type="warning")
        self.drop_button_tab1.on_click(self.remove_node_tab1)
        self.remove_unattached_button = Button(label="Remove unattached nodes", button_type="success")
        self.remove_unattached_button.on_click(self.remove_unbound_nodes)        
        self.update_props_button = Button(label="Update Properties", button_type="warning")
        self.update_props_button.on_click(self.update_properties)
        self.update_layout_button = Button(label="Update Layout", button_type="success")
        self.update_layout_button.on_click(self.update_layout)
        self.select_centrality = Select(title="Centrality Metric:", value="Degree Centrality",
                           options=list(self.centrality_metrics.keys()))
        self.select_centrality.on_change('value', self.update_centrality)
        self.slider = Slider(start=0, end=10, value=0, step=1, title="Threshold %")
        self.slider.on_change('value',self.filter_threshold)
        self.slider.value=thresh_val
        #self.filter_threshold('',0,3)

    def reinit(self,network_file,layout_file,count_path,title):
        lines_source=self.network_lines_tab1
        nodes_source=self.nodes_sources_tab1
        self.network_file=network_file
        self.layout_file=layout_file
        self.count_path=count_path
        self.network_plots_n_circle_tab1[0].title.text=title
        self.network_tuple = self.load_network(network_file,layout_file)
        network,layout=self.network_tuple
        print('loaded new network')
        nodes, nodes_coordinates = zip(*sorted(layout.items()))
        count_dict= dict(pickle.load(open(self.count_path,'rb')))
        nodes_xs, nodes_ys = list(zip(*nodes_coordinates))
        node_occurances = [count_dict[node] for node in nodes]
        nodes_source.data['x'] = nodes_xs
        nodes_source.data['y'] = nodes_ys
        nodes_source.data['name'] = nodes
        nodes_source.data['counts'] = node_occurances
        lines_source.data = self.get_edges_specs(network, layout)
        self.update_properties()
        self.slider.value=8
        self.filter_threshold('',0,8)
        
    def load_network(self,network_file,layout_file):
            network = pickle.load(open(network_file,'rb'))
            layout =pickle.load(open(layout_file,'rb'))
            return (network,layout)
    


    def column_source(self,layout,count_path):
        nodes, nodes_coordinates = zip(*sorted(layout.items()))
        count_dict= dict(pickle.load(open(count_path,'rb')))
        nodes_xs, nodes_ys = list(zip(*nodes_coordinates))
        node_occurances = [count_dict[node] for node in nodes]
        nodes_source = ColumnDataSource(dict(x=nodes_xs, y=nodes_ys,
                                     name=nodes,counts=node_occurances))       
        return nodes_source



    def create_network_plot(self,nodes_source,title='',width=800):
        plot = figure(plot_width=width, plot_height=700,tools=['tap','box_zoom', 'reset','pan','wheel_zoom'],title=title)
        plot.title.text_font = "helvica"
        plot.title.text_font_style = "bold"
        plot.title.text_font_size = "20px"
        plot.background_fill_color = "beige"
        plot.background_fill_alpha = 0.2
        g1 = Circle(x='x', y='y',size=2,fill_color='blue')
        g1_r = plot.add_glyph(source_or_glyph=nodes_source, glyph=g1)
        g1_hover = HoverTool(renderers=[g1_r],
                             tooltips=[('name', '@name'), ('count', '@counts')])
        glyph_text = Text(x="x", y="y", text="name", text_color="#ff4a4a",text_font_size='6pt',text_alpha=0.7)
    
        plot.add_glyph(nodes_source, glyph_text)
        plot.add_tools(g1_hover)
        plot.grid.grid_line_color=None
        plot.axis.visible = False
        return plot,g1_r,glyph_text



    def get_edges_specs(self,_network, _layout):
        d = dict(xs=[], ys=[], alphas=[])
        weights = [d['weight'] for u, v, d in _network.edges(data=True)]
        max_weight = max(weights)
        calc_alpha = lambda h: 0.1 + 0.6 * (h / max_weight)
    
        for u, v, data in _network.edges(data=True):
            d['xs'].append([_layout[u][0], _layout[v][0]])
            d['ys'].append([_layout[u][1], _layout[v][1]])
            d['alphas'].append(calc_alpha(data['weight']))
        return d

    def add_lines(self,network_tuple,plot):
        lines_source = ColumnDataSource(self.get_edges_specs(*network_tuple))
        r_lines = plot.multi_line('xs', 'ys', line_width=2, alpha='alphas', color='navy',
                              source=lines_source)
        return lines_source

    
    def get_centrality_n_community(self,network,nodes_source,g1_r):
        community_colors=self.community_colors
        centrality = networkx.algorithms.centrality.degree_centrality(network)
        # first element, are nodes again
        _, nodes_centrality = zip(*sorted(centrality.items()))
        nodes_source.add([7 + 10 * t / max(nodes_centrality) for t in nodes_centrality], 'centrality')
    
        partition = community.best_partition(network)
        p_, nodes_community = zip(*sorted(partition.items()))
        nodes_source.add(nodes_community, 'community')
        
        nodes_source.add([community_colors[t % len(community_colors)]\
                      for t in nodes_community], 'community_color')
        g1_r.glyph.size = 'centrality'
        g1_r.glyph.fill_color = 'community_color'
    
    


    def remove_node_1_net(self,nodes_source,lines_source,network,layout):
        print('line 92')
        print(type(nodes_source.selected['1d']['indices']))
        print(len(nodes_source.selected['1d']['indices']))
        if(nodes_source.selected['1d']['indices']):
            idx = nodes_source.selected['1d']['indices'][0]
        else:
            return
        # update networkX network object
        node = nodes_source.data['name'][idx]
        network.remove_node(node)
        print('line 97')
        # update layout
        layout.pop(node)
    
        # update nodes ColumnDataSource
        new_source_data = dict()
        for col in nodes_source.column_names:
            print('line 104')
            new_source_data[col] = [e for i, e in enumerate(nodes_source.data[col]) if i != idx]
        nodes_source.data = new_source_data
    
        # update lines ColumnDataSource
        lines_source.data = self.get_edges_specs(network, layout)

    def remove_node_tab1(self):
        self.remove_node_1_net(self.nodes_sources_tab1,self.network_lines_tab1,*self.network_tuple)



    def remove_unbound_nodes(self):
        network,layout=self.network_tuple
        lines_source=self.network_lines_tab1
        nodes_source=self.nodes_sources_tab1
        unbound_nodes = []
        for node in network.nodes():
            if not network.edges(node):
                unbound_nodes.append(node)
        for node in unbound_nodes:
            network.remove_node(node)
            layout.pop(node)
    
        nodes, nodes_coordinates = zip(*sorted(layout.items()))
        count_dict= dict(pickle.load(open(self.count_path,'rb')))
        nodes_xs, nodes_ys = list(zip(*nodes_coordinates))
        node_occurances = [count_dict[node] for node in nodes]
        nodes_source.data['x'] = nodes_xs
        nodes_source.data['y'] = nodes_ys
        nodes_source.data['name'] = nodes
        nodes_source.data['counts'] = node_occurances
        self.update_properties()
        lines_source.data = self.get_edges_specs(network, layout)

    def update_properties(self):
        community_colors=self.community_colors
        network,layout=self.network_tuple
        nodes_source=self.nodes_sources_tab1
        partition = community.best_partition(network)
        p_, nodes_community = zip(*sorted(partition.items()))

        nodes_source.data['community'] = nodes_community
        nodes_source.data['community_color'] = [community_colors[t % len(community_colors)]
                                                for t in nodes_community]
        centrality = self.centrality_metrics[self.select_centrality.value](network, weight='weight')
        _, nodes_centrality = zip(*sorted(centrality.items()))
        nodes_source.data['centrality'] = [7 + 10 * t / max(nodes_centrality) for t in nodes_centrality]


    def update_centrality(self,attrname, old, new):
        network,_=self.network_tuple
        nodes_source=self.nodes_sources_tab1
        centrality = self.centrality_metrics[self.select_centrality.value](network, weight='weight')
        _, nodes_centrality = zip(*sorted(centrality.items()))
        nodes_source.data['centrality'] = [7 + 10 * t / max(nodes_centrality) for t in nodes_centrality]





    def update_layout(self):
        network,layout=self.network_tuple
        lines_source=self.network_lines_tab1
        nodes_source=self.nodes_sources_tab1
        new_layout = networkx.spring_layout(network, k=1.1/sqrt(network.number_of_nodes()),
                                            iterations=100)
        layout = new_layout
        nodes, nodes_coordinates = zip(*sorted(layout.items()))
        nodes_xs, nodes_ys = list(zip(*nodes_coordinates))
        nodes_source.data['x'] = nodes_xs
        nodes_source.data['y'] = nodes_ys
        lines_source.data = self.get_edges_specs(network, layout)
    


    

    def filter_threshold(self,attrname, old, new):
        network,layout=self.network_tuple
        if(old == new):
            return
        if(old > new):
            self.network_tuple=self.load_network(self.network_file,self.layout_file) 
            network,layout = self.network_tuple
        weights = [d['weight'] for u, v, d in network.edges(data=True)]
        max_weight = max(weights)
        min_weight = min(weights)
        threshold = (new*(max_weight-min_weight)/100.0)
        to_remove_list=[]
        sources_in=set()
        for (u,v,d) in network.edges(data='weight'):
            if(d<threshold):
                if (((u,v,d) in sources_in) or ((v,u,d) in sources_in)):
                    continue
                to_remove_list.append((u,v))
                sources_in.add((u,v,d))
        network.remove_edges_from(to_remove_list)
        self.remove_unbound_nodes()
        font_size=6+new
        font_size=min(10,font_size)
        self.network_plots_n_circle_tab1[2].text_font_size= '{}pt'.format(font_size)
        self.update_layout()

    def return_view(self):
        return column(self.network_plots_n_circle_tab1[0],row(widgetbox(self.slider,self.select_centrality),\
                      widgetbox(self.drop_button_tab1,self.remove_unattached_button),\
                      widgetbox(self.update_props_button, self.update_layout_button,)))



list_sources=['Trump','Black']

            
def update_source(attrname, old, new):
    if(new==old):
        return
    if(new=='Trump'):
        network_blm.reinit(join(data_path_prefix,'network_antimaga'),join(data_path_prefix,'layout_antimaga'),\
                               join(data_path_prefix,'graph_nodes_antimaga'),'Resist')
        network_alllm.reinit(join(data_path_prefix,'network_maga'),join(data_path_prefix,'layout_maga'),\
                                 join(data_path_prefix,'graph_nodes_maga'),'MAGA')
        cloud_blm.reinit(allm_img,'Wordcloud with All Lives Matter Tweets')
        cloud_alllm.reinit(blm_img,'Wordcloud with Black Lives Matter Tweets')
        time_count_plot1.reinit(join(data_path_prefix,'time_occur_maga'),'MAGA',\
                                join(data_path_prefix,'time_occur_antimaga'),'Resist')
        time_polarity_plot1.reinit(join(data_path_prefix,'polarity_maga_df'),'MAGA',\
                                    join(data_path_prefix,'polarity_antimaga_df'),'Resist')
        
        time_subjectivity_plot1.reinit(join(data_path_prefix,'subjectivity_maga_df'),'MAGA',\
                                        join(data_path_prefix,'subjectivity_antimaga_df'),'Resist','Subjectivity')
        
        percent_subjectivity_plot_blm.reinit(join(data_path_prefix,'subjectivity_percent_df_antmaga'),'Subjectivity',\
                                                        'Resist' , join(data_path_prefix,'events_keshav'))
        
        percent_subjectivity_plot_antiblm.reinit( join(data_path_prefix,'subjectivity_percent_df_maga'),'Subjectivity',\
                                                        'MAGA' , join(data_path_prefix,'events_keshav'))
        
        percent_polarity_plot_blm.reinit(join(data_path_prefix,'polarity_percent_df_antmaga'),'Polarity',\
                                                        'Resist' , join(data_path_prefix,'events_keshav'))
        
        percent_polarity_plot_antiblm.reinit(join(data_path_prefix,'polarity_percent_df_maga'),'Polarity',\
                                                        'MAGA' , join(data_path_prefix,'events_keshav'))

    else:
        network_blm.reinit(join(data_path_prefix,'network_blm'),join(data_path_prefix,'layout_blm'),\
                               join(data_path_prefix,'graph_nodes_blm'),'BlackLivesMatter')
        network_alllm.reinit(join(data_path_prefix,'network_antblm'),join(data_path_prefix,'layout_antblm'),\
                                 join(data_path_prefix,'graph_nodes_antblm'),'All Lives Matter')
        
        cloud_blm.reinit(blm_img,'Wordcloud with Black Lives Matter Tweets')
        cloud_alllm.reinit(allm_img,'Wordcloud with All Lives Matter Tweets')
        time_count_plot1.reinit(join(data_path_prefix,'time_occur_blm'),'BLM',\
                                        join(data_path_prefix,'time_occur_alllm'),'ALLLM')
        time_polarity_plot1.reinit(join(data_path_prefix,'polarity_blm_df'),'Black Lives Matter',\
                                        join(data_path_prefix,'polarity_antblm_df'),'All Lives Matter')
        
        time_subjectivity_plot1.reinit(join(data_path_prefix,'subjectivity_blm_df'),'Black Lives Matter',\
                                        join(data_path_prefix,'subjectivity_antblm_df'),'All Lives Matter','Subjectivity')
        
        
        percent_subjectivity_plot_blm.reinit(join(data_path_prefix,'subjectivity_percent_df_blm'),'Subjectivity',\
                                                        'Black Lives Matter' , join(data_path_prefix,'events_ada'))
        
        percent_subjectivity_plot_antiblm.reinit(join(data_path_prefix,'polarity_percent_df_antblm'),'Subjectivity',\
                                                        'All Lives Matter' , join(data_path_prefix,'events_ada'))
        
        percent_polarity_plot_blm.reinit(join(data_path_prefix,'polarity_percent_df_blm'),'Polarity',\
                                                        'Black Lives Matter' , join(data_path_prefix,'events_ada'))
        
        percent_polarity_plot_antiblm.reinit(join(data_path_prefix,'polarity_percent_df_antblm'),'Polarity',\
                                                        'All Lives Matter' , join(data_path_prefix,'events_ada'))

    
   
              
select_data_source = Select(title="Select Source", value=list_sources[0],
                           options=list(list_sources))

select_data_source.on_change('value', update_source)



data_path_prefix=join(dirname(__file__), 'data')


allm_img="/ada_app_2017/static/images/cloud_alllm.png"
blm_img="/ada_app_2017/static/images/cloud_blm.png"

static_path = '/ada_app_2017/static/{}'

time_count_plot1=Create_time_count_plot(join(data_path_prefix,'time_occur_antimaga'),'Resist',\
                                        join(data_path_prefix,'time_occur_maga'),'MAGA')

time_count_plot2=Create_time_count_plot(join(data_path_prefix,'time_occur_misogyny'),'Misogyny')

time_polarity_plot1=Create_polarity_plot(join(data_path_prefix,'polarity_antimaga_df'),'Resist',\
                                        join(data_path_prefix,'polarity_maga_df'),'MAGA')

time_polarity_plot2=Create_polarity_plot(join(data_path_prefix,'polarity_miso_df'),'Misogyny')

time_subjectivity_plot1=Create_polarity_plot(join(data_path_prefix,'subjectivity_antimaga_df'),'Resist',\
                                        join(data_path_prefix,'subjectivity_maga_df'),'MAGA','Subjectivity','Subjectivity')

time_subjectivity_plot2=Create_polarity_plot(join(data_path_prefix,'subjectivity_miso_df'),'Misogyny',title='Subjectivity',y_axis='Subjectivity')


percent_subjectivity_plot_blm=Create_percent_sentiment_plot( join(data_path_prefix,'subjectivity_percent_df_antmaga'),'Subjectivity',\
                                                        'Resist' , join(data_path_prefix,'events_keshav'))

percent_subjectivity_plot_antiblm=Create_percent_sentiment_plot( join(data_path_prefix,'subjectivity_percent_df_maga'),'Subjectivity',\
                                                        'MAGA' , join(data_path_prefix,'events_keshav'))

percent_subjectivity_plot_miso=Create_percent_sentiment_plot( join(data_path_prefix,'subjectivity_percent_df_miso'),'Subjectivity',\
                                                        'Misogyny' ,width=1600)

lda_vis = Create_LDA_html( join(data_path_prefix,'lda_vis_maga.html'),'Topic visualization')

percent_polarity_plot_blm=Create_percent_sentiment_plot( join(data_path_prefix,'polarity_percent_df_antmaga'),'Polarity',\
                                                        'Resist' , join(data_path_prefix,'events_keshav'))

percent_polarity_plot_antiblm=Create_percent_sentiment_plot( join(data_path_prefix,'polarity_percent_df_maga'),'Polarity',\
                                                        'MAGA' , join(data_path_prefix,'events_keshav'))

percent_polarity_plot_miso=Create_percent_sentiment_plot( join(data_path_prefix,'polarity_percent_df_miso'),'Polarity',\
                                                        'Misogyny',width=1600)


network_blm=Create_network(join(data_path_prefix,'network_maga'),join(data_path_prefix,'layout_maga'),\
                           join(data_path_prefix,'graph_nodes_maga'),'MAGA')

network_alllm=Create_network(join(data_path_prefix,'network_antimaga'),join(data_path_prefix,'layout_antimaga'),\
                             join(data_path_prefix,'graph_nodes_antimaga'),'Resist')

network_miso=Create_network(join(data_path_prefix,'network_misogyny'),join(data_path_prefix,'layout_misogyny'),\
                           join(data_path_prefix,'graph_nodes_misogyny'),'Misogyny',1600,3)

net_row=row(network_blm.return_view(),network_alllm.return_view())

cloud_blm=Create_image_html(blm_img,'Wordcloud with Black Lives Matter Tweets')

cloud_alllm=Create_image_html(allm_img,'Wordcloud with All Lives Matter Tweets')

cloud_row=row(cloud_blm.return_view(),cloud_alllm.return_view())

percent_subj_row=row(percent_subjectivity_plot_blm.return_view(),percent_subjectivity_plot_antiblm.return_view())

percent_polar_row= row(percent_polarity_plot_blm.return_view(),percent_polarity_plot_antiblm.return_view())

tab_blm = Panel(child=column(widgetbox(select_data_source),time_count_plot1.return_view()\
                             ,net_row,lda_vis.return_view(),time_polarity_plot1.return_view(),time_subjectivity_plot1.return_view(),percent_subj_row,percent_polar_row),title='Comparisons')

tab_miso = Panel(child=column(time_count_plot2.return_view(),network_miso.return_view(),time_subjectivity_plot2.return_view(),time_polarity_plot2.return_view(),
                              percent_subjectivity_plot_miso.return_view(),percent_polarity_plot_miso.return_view()),title='Solo Analysis')


tabs = Tabs(tabs=[tab_blm,tab_miso])

curdoc().add_root(tabs)
curdoc().title = "ADA project"
    
