# 作者：周毅
# 开发时间： 2022/9/2 9:48
import numpy as np
import pandas as pd
import math
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from tqdm.notebook import tqdm, trange
import mesa
from optimizedModel import Market
from mesa.visualization.ModularVisualization import VisualizationElement, CHART_JS_FILE,ModularServer,SocketHandler
from multiprocessing import freeze_support
import json
import tornado.autoreload
import tornado.ioloop
import tornado.web
import tornado.websocket
import tornado.escape
import tornado.gen
import webbrowser




class HistogramModule(VisualizationElement):
    package_includes = [CHART_JS_FILE]
    local_includes = ["HistogramModule.js"]
    def __init__(self,bins, canvas_height, canvas_width, p, type):
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.bins = bins
        self.p = p
        self.type = type
        # self.type = type
        # print(type)
        new_element = "new HistogramModule({}, {}, {}, {})"
        new_element = new_element.format(bins,canvas_width, canvas_height, '"'+type+'"')
        self.js_code = "elements.push(" + new_element + ");"


    def render(self, model):
        if(self.type == 'histogram_h'):
            wealth = [agent['wealth'] for agent in model.agents if agent['trader_type']==1]
        else:
            wealth = [agent['wealth'] for agent in model.agents if agent['trader_type']==0]
        # print("策划")
        # print(self.p)
        hist = np.histogram(wealth, bins=self.bins,range=(0,self.p))[0]
        # print(hist)

        return [int(x) for x in hist]

#重新写chartModel的方法，主要是修改了一些render的val值，用于初始化图表的起点
class chartModel(VisualizationElement):
    package_includes = [CHART_JS_FILE, "ChartModule.js"]
    def __init__(
            self,
            series,
            canvas_height=200,
            canvas_width=500,
            data_collector_name="datacollector",
            val = 10
    ):
        self.series = series
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.data_collector_name = data_collector_name
        self.val = val

        series_json = json.dumps(self.series)
        new_element = "new ChartModule({}, {},  {})"
        new_element = new_element.format(series_json, canvas_width, canvas_height)
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        current_values = []
        data_collector = getattr(model, self.data_collector_name)
        for s in self.series:
            name = s["Label"]
            try:
                val = data_collector.model_vars[name][-1]  # Latest value
            except (IndexError, KeyError):
                val = self.val
            current_values.append(val)
        return current_values


chart = chartModel([{"Label": "last_price","Color": "Black"}],
                    data_collector_name='datacollector',val=10)


chart2 = mesa.visualization.ChartModule([{"Label": "bid_sum",
                      "Color": "red"},{"Label": "ask_sum",
                      "Color": "green"}],
                    data_collector_name='datacollector')


chart3 = chartModel([{"Label": "trade_count",
                      "Color": "green"}],
                    data_collector_name='datacollector',val=10)

chart4 = mesa.visualization.ChartModule([{"Label": "volume",
                      "Color": "yellow"}],
                    data_collector_name='datacollector')


# chart3 = mesa.visualization.ChartModule([{"Label": "compute_cash",
#                       "Color": "yellow"}],
#                     data_collector_name='datacollector')


histogram_l = HistogramModule(list(range(0,20000,500)), 200, 500,500, 'histogram_l')

histogram_h = HistogramModule(list(range(98000000,102000000,50000)), 200, 500,50000,'histogram_h')


# server = mesa.visualization.ModularServer(Market,
#                        [chart,chart2,chart3,],
#                        "Market",
#                        {"N":400, "init_price":10})




class CustomerModularServer(ModularServer):
    def launch(self, port=None, open_browser=True,url="http://127.0.0.1"):
        print(url)
        """Run the app."""
        if port is not None:
            self.port = port
        url = f"{url}:{self.port}"
        print(f"Interface starting at {url}")
        self.listen(self.port)
        if open_browser:
            webbrowser.open(url)
        tornado.autoreload.start()
        tornado.ioloop.IOLoop.current().start()


# server = CustomerModularServer(Market,
#                        [],
#                        "Market",
#                        {"N":400, "init_price":20})



server = mesa.visualization.ModularServer(Market,
                       [chart,chart2,chart3,chart4,histogram_h,histogram_l],
                       "Market",
                       {"N":10000, "init_price":10})





# freeze_support()
# results = batch_run(
#     MoneyModel,
#     parameters=params,
#     iterations=5,
#     max_steps=100,
#     number_processes=None,
#     data_collection_period=1,
#     display_progress=True,
# )
server.port = 8521 # The default
server.launch()
