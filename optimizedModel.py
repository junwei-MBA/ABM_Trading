import numpy as np
import pandas as pd
import math
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from tqdm.notebook import tqdm, trange

pd.options.mode.chained_assignment = None  # default='warn'

agent_number=10000
steps=4000
# np.random.seed(0) #打开的时候每次使用相同的随机数，便于调试
def createAgent(unique_id,initial_invest,initial_price):
    dic={}
    dic["id"]=unique_id
    dic['unique_id']=unique_id
    dic["initial_invest"]=initial_invest
    dic["trader_type"]=0
    dic["status"]=0
    dic["wealth"]=initial_invest
    dic["available_cash"]=initial_invest
    dic["balance"]=initial_invest
    dic["position"]=0
    dic["weighted_cost"]=0.0
    dic["order_qty"]=0
    dic["flag"]=0
    dic["bid_size_series"]=[]
    dic["buy_size_series"]=[]
    dic["ask_size_series"]=[]
    dic["sell_size_series"]=[]
    dic["n1"]=np.random.normal(0,0.3)
    dic["n2"]=np.random.normal(0,0.6)
    dic["n3"]=np.random.normal(0,0.1)
    dic["lvalue"]=int(np.random.uniform(1,30))
    dic["buy_trades"]=0
    dic["sell_trades"]=0
    dic["fee"]=0.0
    dic["close_profit"]=0
    dic["ask_price"]=0
    dic["bid_price"]=0
    dic["expect_price"]=0
    dic["price_flu"]=0
    dic["EDI"]=0.0
    dic["guess"]=round(np.random.normal(initial_price,2),2) 
    dic["step_number"]=0
    dic["expect_return"]=0.0
    dic["bid_size"]=0
    dic["ask_size"]=0
    dic["direction"]=-1
    dic["max_order_qty"]=0
    dic["ask_qty"]=0
    dic["bid_qty"]=0
    return dic
templateAgent=createAgent(0,0.0,0.0)
keptKeys=['id','trader_type','wealth','available_cash','balance','position','buy_trades','sell_trades','EDI','step_number']
templateAgent=createAgent(0,0,0)
for i in list(templateAgent.keys()):
    if(i not in keptKeys):
        templateAgent.pop(i)
#预先创建好agentsData，内存也分配好，这样在添加元素的时候不会产生拓展数组的操作
agentsData=np.tile(list(templateAgent.values()),(steps*agent_number,1)) 

def collectAgents(step,agents):
    global agentsData
    base=(step-1)*agent_number
    for i in range(len(agents)):
        targetAgent=agents[i]
        agentsData[base+i][0]=targetAgent['id']
        agentsData[base+i][1]=targetAgent['trader_type']
        agentsData[base+i][2]=targetAgent['wealth']
        agentsData[base+i][3]=targetAgent['available_cash']
        agentsData[base+i][4]=targetAgent['balance']
        agentsData[base+i][5]=targetAgent['position']
        agentsData[base+i][6]=targetAgent['buy_trades']
        agentsData[base+i][7]=targetAgent['sell_trades']
        agentsData[base+i][8]=targetAgent['EDI']
        agentsData[base+i][9]=targetAgent['step_number']
def stepAgent(agent):
    agent['step_number']+=1
    if agent['trader_type']==0:
        agent["bid_size_series"].append(agent["bid_size"])
        agent["ask_size_series"].append(agent["ask_size"])
        agent["buy_size_series"].append(agent["buy_trades"])
        agent["sell_size_series"].append(agent["sell_trades"])
        if len(agent["bid_size_series"])>30:
            del agent["bid_size_series"][:-20]
            del agent["ask_size_series"][:-20]
            del agent["buy_size_series"][:-20]
            del agent["sell_size_series"][:-20]
class Market(Model):
    def __init__(self, N,init_price):
        #init_price=10
        self.initial_invest=10000
        self.number_agents = N
        self.stock_price=[init_price]
        self.step_count=0
        self.running=True
        self.close_price=[]
        self.agents=[]
        self.margin_rate=0.15 # 保证金，按照持仓金额的15%计算
        order_que={'ID': [],'position':[],'guess':[],'OrderQty':[],'direction':[],'status':[],'type':[],'time': []}
        self.ask_df=pd.DataFrame(order_que)
        self.bid_df=pd.DataFrame(order_que)
        self.bid_sum=0
        self.ask_sum=0
        self.current_price=None # this is the close price for each step
        # Create agents
        for i in range(self.number_agents):
            a = createAgent(i,self.initial_invest,init_price)
            self.agents.append(a)
        
        for agent in self.agents:
            #这里设置low frequency trader的比例为0.98，即98%
            agent["trader_type"]=np.random.choice([0,1],replace=True,p=[0.98,0.02])
            if agent['trader_type']==1:
                agent["wealth"]=10000*agent['initial_invest']
                agent["balance"]=agent["wealth"]
                agent['available_cash']=10000*agent['initial_invest']
        #record_df= {'time': [],'price': [],'buyer':[],'seller':[],'qty':[]}
        #self.records=pd.DataFrame(record_df) #不再在每步记录此前所有的交易记录
        self.datacollector = DataCollector(
            model_reporters={"time":"step_count","last_price": "current_price","volume":"step_volume","trade_count":"trade_count","bid_sum":"bid_sum","ask_sum":"ask_sum"},
        ) # buy_trades 是买入成交的，sell_trades是卖出成交的，高频交易者很可能两者同时都有，因此分开了
    def step(self):
        self.step_count += 1
        self.step_volume=0
        self.trade_count=0
        n1s=np.random.normal(0,0.3,self.number_agents)
        n2s=np.random.normal(0,0.6,self.number_agents)
        n3s=np.random.normal(0,0.1,self.number_agents)
        lvalues=np.random.uniform(1,30,self.number_agents).astype(np.int32)
        
        p_low_ap_randoms_one=np.random.uniform(0.01,0.1,self.number_agents)
        p_low_ap_randoms_two=np.random.uniform(p_low_ap_randoms_one,0.1,self.number_agents)
        p_low_ap_randoms_three=np.random.uniform(0.01,p_low_ap_randoms_one,self.number_agents)
        random_four=np.random.uniform(0,1,self.number_agents)
        lft_low_ap_randoms=np.random.rand(self.number_agents)
        
        for index in range(self.number_agents):
            agent=self.agents[index]
        
            if agent['wealth']<=0:
                agent['status']=-1 # agent已经被淘汰了
            else:
                if agent['trader_type']==0:
                    #这里设置low frequency trader用户中活跃的比例,该比例为在0.01到0.1之间均匀分布的随机值，然后每30步评估一次并更新
                    p_low_ap=p_low_ap_randoms_one[index]
                    if self.step_count<60:
                        p_low_ap=p_low_ap
                    elif self.step_count % 30 ==0:
                        if agent['wealth']>agent['initial_invest']:
                            agent['flag']=1
                            p_low_ap=p_low_ap_randoms_two[index]
                        else:
                            agent['flag']=-1
                            p_low_ap=p_low_ap_randoms_three[index]
                            if random_four[index]<0.3:
                                agent['n1']=n1s[index]
                                agent['n2']=n2s[index]
                                agent['n3']=n3s[index]
                                agent['lvalue']=lvalues[index]
                    else:
                        if agent['flag']==1:
                            p_low_ap=p_low_ap_randoms_two[index]
                        else:
                            p_low_ap=p_low_ap_randoms_three[index]
#                     agent['status']=rng.choice([1,0],replace=True,p=[p_low_ap,1-p_low_ap])
                    agent['status']=0 if lft_low_ap_randoms[index]>p_low_ap else 1
                else: #这里设置 high frequency trader用户的活跃比例
                    agent['status']=0 #先设定为不活跃，如果满足下面的条件改成活跃
                    if len(self.close_price)>2:
                        p=self.close_price
                        p_t_flu=abs((p[-2]-p[-3])/p[-2])*10000
                        agent['price_flu']=p_t_flu
                        if p_t_flu>5:
                            agent['status']=1  
        ask_df,bid_df=self._getguess()
        #在最初的32步里，我们只是为了获取市场初始的价格序列，所以即使非活跃用户也可以交易，而且交易量大部分是0，不影响各自的持仓和财富
        self.ask_df=pd.concat([self.ask_df,ask_df],ignore_index = True, axis = 0)
        self.bid_df=pd.concat([self.bid_df,bid_df],ignore_index = True, axis = 0)
        #如果不用ID排序，那么HFT在价格相同的时候就是随机的成交顺序，注意在guess那里也有ID排序，要一起去掉
        self.ask_df.sort_values(by=["guess","time","type","ID"], inplace=True, ascending=[True,True,False,True])
        #self.ask_df.sort_values(by=["guess","time","type"], inplace=True, ascending=[True,True,False]) 
        
        self.bid_df.sort_values(by=["guess","time","type","ID"], inplace=True, ascending=[False,True,False,True])
        #self.bid_df.sort_values(by=["guess","time","type"], inplace=True, ascending=[False,True,False])
       
        #超过33步以后，只许活跃用户交易
        if self.step_count>33:
            self.ask_df=self.ask_df[self.ask_df['status']==1]
            self.bid_df=self.bid_df[self.bid_df['status']==1]
        
        self.ask_df=self.ask_df.loc[(self.ask_df['type']==1)&(self.ask_df['time']>(self.step_count-1))|(self.ask_df['type']==0)&(self.ask_df['time']>(self.step_count-10))]
        self.bid_df=self.bid_df.loc[(self.bid_df['type']==1)&(self.bid_df['time']>(self.step_count-1))|(self.bid_df['type']==0)&(self.bid_df['time']>(self.step_count-10))]
        self.ask_df = self.ask_df.reset_index(drop=True)
        self.bid_df = self.bid_df.reset_index(drop=True)
        self.bid_sum=self.bid_df['OrderQty'].sum()
        self.ask_sum=self.ask_df['OrderQty'].sum()
        self.trade()
    def trade(self):
        if self.ask_df.shape[0]>0:
#             self.ask_df,self.bid_df=self._trade(self.ask_df,self.bid_df)
            self.ask_df,self.bid_df=self.newTrade(self.ask_df,self.bid_df)
        else:
            if self.current_price is not None:
                self.close_price.append(self.current_price)
        self._getWealth()
        self.datacollector.collect(self)
        collectAgents(self.step_count,self.agents)
        for agent in self.agents:
            stepAgent(agent)
        # self.schedule.step()    
    def _getguess(self):
        #把每个agent的guess数值以及agent的ID和持仓信息放到dataframe ，随机或者根据持仓做出方向选择，最后返回dataframe
        guess_list=[]
        p=self.close_price
        market_report_last_10step = self.datacollector.get_model_vars_dataframe()[-10:-1]
       
        market_bid_sum_last_10step=market_report_last_10step['bid_sum'].sum()
        market_ask_sum_last_10step=market_report_last_10step['ask_sum'].sum()
        volume_sum_last_10step=market_report_last_10step['volume'].sum()
        
        
        random_choices=np.random.choice([1,-1],self.number_agents)
        uniforms_one=np.random.uniform(0,10,self.number_agents)
        normals_one=np.random.normal(0,1,self.number_agents)
        uniforms_k1=np.random.uniform(-0.002,0.01,self.number_agents)
        uniforms_ital=np.random.uniform(200,1000,self.number_agents)
        uniforms_ital2=np.random.uniform(uniforms_ital,1000,self.number_agents)
        uniforms_ital3=np.random.uniform(200,uniforms_ital,self.number_agents)
        #uniforms_itah=np.random.uniform(0.1,0.5,self.number_agents)
        uniforms_itah=np.random.uniform(0.001,0.005,10000)
        if (market_bid_sum_last_10step+market_ask_sum_last_10step)>0:
            market_order_exe_rate=volume_sum_last_10step/(market_bid_sum_last_10step+market_ask_sum_last_10step)
        else:
            market_order_exe_rate=0
        net_threshold=1000*volume_sum_last_10step/10 # this is used for HFT to control their inventory
#         print("net threshold is %s"%(net_threshold))
        for agent in self.agents:
            if agent['available_cash']<=0: #如果可用资金小于等于零，强制平仓到可用资金为正
                if agent['position'] > 0:
                    agent['ask_size']=int(abs(agent['available_cash'])/self.margin_rate/self.current_price)+1
                    agent['ask_price']=self.current_price*0.5  # 打5折保证出价足够低，便于强平交易优先成交
                    agent['direction']=-1 #表示卖出
                    guess_list.append([agent['unique_id'],agent['position'],agent['ask_price'],agent['ask_size'],agent['direction'],agent['status'],agent['trader_type']])
                else:
                    #持有空仓强平的时候需要买入
                    agent['bid_size']=int(abs(agent['available_cash'])/self.margin_rate/self.current_price)+1
                    agent['bid_price']=self.current_price*2 # 同上，保证可以优先买入
                    agent['direction']=1 #表示买入
                    guess_list.append([agent['unique_id'],agent['position'],agent['bid_price'],agent['bid_size'],agent['direction'],agent['status'],agent['trader_type']])
            else: #这时候可用资金大于0，按照下面的方法报单
                agent['bid_size']=0
                agent['ask_size']=0
                agent['buy_trades']=0
                agent['sell_trades']=0
                agent['fee']=0
                agent['close_profit']=0
                if agent['trader_type']==1:
                    if net_threshold>0:
                        agent['EDI']=round(abs(agent['position']/net_threshold),3)
                if agent['status']==0:
                    #不活跃用户，可能是LFT也可能是 HFT
                    agent['order_qty']=0
                    agent['direction']=random_choices[agent['id']]
                    #if agent.guess is not None:
                    guess_list.append([agent['unique_id'],agent['position'],agent['guess'],agent['order_qty'],agent['direction'],agent['status'],agent['trader_type']])
                elif agent['trader_type']==0:
                    #LFT 且活跃
                    if len(self.close_price)>32 and int(uniforms_one[agent['id']])==5: # 这样使得LFT出价的概率是十分之一
                        sum2=0
                        l=agent['lvalue']
                        n1=agent['n1']
                        n2=agent['n2']
                        n3=agent['n3']

                        pf=10 # foundamental value 取值50参考的文献 Yibing Xiong, Takao Terano 2015
                        for j in range(1, l):
                            sum2+=math.log(p[-2]/p[-2-j])

                        agent['expect_return']=n1*math.log(pf/p[-1])+n2*sum2/l+n3*normals_one[agent['id']]
                        agent['expect_price']=p[-1]*math.exp(agent['expect_return'])
                        #define kl as price fluctuation parameter, see reference paper for details
                        kl=uniforms_k1[agent['id']]
                        agent['ask_price']=round(p[-2]*(1-kl),2)
                        agent['bid_price']=round(p[-2]*(1+kl),2)
                        #define ital as size fluctuation parameter of LFT,see reference for detail
                        
                        if self.step_count<60: #前33步没有真正成交，所以第30步不需要评估，从第60步开始评估
                            ital=uniforms_ital[agent['id']]
                        else:
                            if agent['flag']==1:
                                ital=uniforms_ital2[agent['id']]
                            else:
                                ital=uniforms_ital3[agent['id']]

                        if agent['ask_price'] > agent['expect_price']:
                            agent['guess']=agent['ask_price']
                            agent['order_qty']=int(abs(agent['expect_return']*ital))
                            agent['max_order_qty']=int(agent['available_cash']/self.margin_rate/agent['guess'])
                            agent['order_qty']=min(agent['order_qty'],agent['max_order_qty'])
                            agent['direction']=-1 # -1表示卖出
                            agent['ask_size']=agent['order_qty']
                            guess_list.append([agent['unique_id'],agent['position'],agent['guess'],agent['order_qty'],agent['direction'],agent['status'],agent['trader_type']])
                        if agent['bid_price'] < agent['expect_price']:
                            agent['guess']=agent['bid_price']
                            agent['order_qty']=int(abs(agent['expect_return']*ital))
                            agent['max_order_qty']=int(agent['available_cash']/self.margin_rate/agent['guess'])
                            agent['order_qty']=min(agent['order_qty'],agent['max_order_qty'])
                            agent['direction']=1  # 1 表示买入
                            agent['bid_size']=agent['order_qty']
                            guess_list.append([agent['unique_id'],agent['position'],agent['guess'],agent['order_qty'],agent['direction'],agent['status'],agent['trader_type']])

                else:
                    #HFT 且活跃
                    #define kh as HFT　order price fluctuation, itah as HFT order absorption rate 
                    if self.bid_df is not None and self.ask_df is not None:


                        qb=self.bid_df['OrderQty'].sum()
                        qs=self.ask_df['OrderQty'].sum()
                        kh=0.01
                        itah=uniforms_itah[agent['id']]

                        agent['ask_qty']=int(0.5*(qb+qs)*itah)
                        agent['bid_qty']=int(0.5*(qb+qs)*itah)
                        agent['ask_price']=round(p[-1]+kh,2)
                        agent['bid_price']=round(p[-1]-kh,2)
                        if self.step_count>45:
                            buy_trades_last_10step=sum(agent['buy_size_series'][-10:])
                            sell_trades_last_10step=sum(agent['sell_size_series'][-10:])
                            bid_sum_last_10step=sum(agent['bid_size_series'][-10:])
                            ask_sum_last_10step=sum(agent['ask_size_series'][-10:])
                            if bid_sum_last_10step+ask_sum_last_10step>0:
                                self_order_exe_rate=(buy_trades_last_10step+sell_trades_last_10step)/(bid_sum_last_10step+ask_sum_last_10step)
                            else:
                                self_order_exe_rate=0
                            if self_order_exe_rate*market_order_exe_rate>0:
                                if abs(qb-qs)/(qb+qs)<0.5 and (qb+qs)>0: # HFT will use passive market making
                                    agent['ask_qty']=min(qb,qs)*0.5*(self_order_exe_rate+market_order_exe_rate)
                                    agent['bid_qty']=min(qb,qs)*0.5*(self_order_exe_rate+market_order_exe_rate)
                                else: # HFT will use aggressive market making
                                    agent['ask_qty']=abs(qb-qs)*0.5*(self_order_exe_rate+market_order_exe_rate)
                                    agent['bid_qty']=abs(qb-qs)*0.5*(self_order_exe_rate+market_order_exe_rate)
                            if abs(agent['position'])<0.5*net_threshold:
                                agent['ask_price']=round(p[-1]+kh,2)
                                agent['bid_price']=round(p[-1]-kh,2)
                            elif abs(agent['position'])<net_threshold:
                                agent['ask_price']=round(p[-1]+2*kh,2)
                                agent['bid_price']=round(p[-1]-2*kh,2)
                            else: #这时净持仓超过了net_threshold
                                if agent['position']>0: #如果是正的，表明是多仓，需要停止买入
                                    agent['bid_qty']=0
                                    agent['bid_price']=0
                                    agent['ask_price']=round(p[-1]+kh,2)
                                else:  #反之是负数，需要停止卖出
                                    agent['ask_qty']=0
                                    agent['ask_price']=0
                                    agent['bid_price']=round(p[-1]-kh,2)

                            agent['ask_qty']=int(min(agent['ask_qty'],net_threshold,agent['available_cash']/self.margin_rate/agent['ask_price'])) # HFT的下单数最大不超过阈值
                            agent['bid_qty']=int(min(agent['bid_qty'],net_threshold,agent['available_cash']/self.margin_rate/agent['bid_price']))
                            # 需要加上手数的限制，不超过资金可开仓的上限
                        agent['ask_size']=agent['ask_qty']
                        agent['bid_size']=agent['bid_qty']
                        if agent['ask_price'] >0:
                            guess_list.append([agent['unique_id'],agent['position'],agent['ask_price'],agent['ask_qty'],-1,agent['status'],agent['trader_type']])
                        if agent['bid_price'] >0:
                            guess_list.append([agent['unique_id'],agent['position'],agent['bid_price'],agent['bid_qty'],1,agent['status'],agent['trader_type']])
                
        
        guess_array=np.array(guess_list)
        df=pd.DataFrame(guess_array)
        df.columns=['ID','position','guess','OrderQty','direction','status','type']
        df['time']=self.step_count
     
            
        df['guess'].replace('', np.nan, inplace=True)
        df.dropna(subset=['guess'], inplace=True)
        #df.sort_values(by="guess", inplace=True, ascending=True)
               

        bid_df=df[df['direction']==1]
        #如果排序不加上ID，那么HFT是按照随机的顺序提交订单，注意在step里还有一次排序，要一起修改
        bid_df.sort_values(by=["guess","type","ID"], inplace=True, ascending=[False,False,True])
        #bid_df.sort_values(by=["guess","type"], inplace=True, ascending=[False,False])
      
        bid_df = bid_df.reset_index(drop=True)

        ask_df=df[df['direction']==-1]
        ask_df.sort_values(by=["guess","type","ID"], inplace=True, ascending=[True,False,True])
        #ask_df.sort_values(by=["guess","type"], inplace=True, ascending=[True,False])

        ask_df = ask_df.reset_index(drop=True)
        
       
        return ask_df,bid_df
    def _getWealth(self):
        for agent in self.agents:
            agent['balance']=agent['balance']+agent['close_profit']-agent['fee']
            temp_profit=agent['position']*(self.current_price-agent['weighted_cost'])
            agent['wealth']=agent['balance']+temp_profit
            # agent['wealth']=agent['wealth']+(agent['position'])*(self.current_price-agent['weighted_cost'])-agent['fee']+agent['close_profit']
            agent['available_cash']=agent['wealth']-abs(agent['position'])*self.current_price*self.margin_rate
    def newTrade(self,ask_df,bid_df):
        ask_series=ask_df.values
        bid_series=bid_df.values
        ask_index=0
        bid_index=0
        ask_len=ask_df.shape[0]
        bid_len=bid_df.shape[0]
        while(ask_index< ask_len and bid_index <bid_len):
            if(ask_series[ask_index][2]<=bid_series[bid_index][2]):
                #如果卖价低于或者等于买价，则可以成交
                volume=0
                buyer=self.agents[int(bid_series[bid_index][0])]
                seller=self.agents[int(ask_series[ask_index][0])]
                last_price=self.stock_price[-1]
                match_price=0
                if(ask_series[ask_index][3]<bid_series[bid_index][3]):
                    #卖手数低于买手数
                    volume=ask_series[ask_index][3]
                    bid_series[bid_index][3]-=volume
                    ask_series[ask_index][3]=0 #把ask的quantity置于0
                    match_price=np.median([ask_series[ask_index][2],bid_series[bid_index][2],last_price])
                    ask_index+=1 #推进ask index
                elif(ask_series[ask_index][3]==bid_series[bid_index][3]):
                    volume=ask_series[ask_index][3]
                    bid_series[bid_index][3]=0 #把ask的quantity置于0
                    ask_series[ask_index][3]=0 #把ask的quantity置于0
                    match_price=np.median([ask_series[ask_index][2],bid_series[bid_index][2],last_price])
                    ask_index+=1 #推进ask index
                    bid_index+=1 #推进bid index
                elif(ask_series[ask_index][3]>bid_series[bid_index][3]):
                    #卖手数高于买手数
                    volume=bid_series[bid_index][3]
                    bid_series[bid_index][3]=0 #把bid的quantity置于0
                    ask_series[ask_index][3]-=volume 
                    match_price=np.median([ask_series[ask_index][2],bid_series[bid_index][2],last_price])
                    bid_index+=1 #推进bid index
                #设置市场价格
                if(match_price>0):
                    self.stock_price.append(match_price)
                self.step_volume+=volume
                if self.step_count>33:
                    self.trade_count+=1
                #设置两个agent的数据
                buyer['buy_trades']+=volume
                seller['sell_trades']+=volume
                buyer['fee']+=volume*match_price*(0.0003-buyer['trader_type']*0.0002)
                seller['fee']+=volume*match_price*(0.0003-seller['trader_type']*0.0002)
#                 if buyer['trader_type']==0:
#                     buyer['fee']+=volume*match_price*0.0003
#                 else:
#                     buyer['fee']+=volume*match_price*0.0001
#                 if seller['trader_type']==0:
#                     seller['fee']+=volume*match_price*0.0003
#                 else:
#                     seller['fee']+=volume*match_price*0.0001
                buyer['position'] += volume
                if buyer['position']!=0:
                    buyer['weighted_cost']=(buyer['weighted_cost']*(buyer['position']-volume)+volume*match_price)/buyer['position']
                else:
                    buyer['close_profit']+=volume*(match_price-buyer['weighted_cost'])
                    buyer['weighted_cost']=0
                seller['position'] -= volume
                if seller['position']!=0:
                    seller['weighted_cost']=(seller['weighted_cost']*(seller['position']+volume)-volume*match_price)/seller['position']
                else:
                    seller['close_profit']+=volume*(match_price-seller['weighted_cost'])
                    seller['weighted_cost']=0
                last_price=match_price
            else:
                #卖价高于了买价了，无法成交
                break
        self.current_price=self.stock_price[-1]
        self.close_price.append(self.current_price)
        new_ask_df=pd.DataFrame(ask_series,columns=["ID",'position','guess','OrderQty','direction','status','type','time'])
        new_bid_df=pd.DataFrame(bid_series,columns=["ID",'position','guess','OrderQty','direction','status','type','time']) 
        new_ask_df=new_ask_df[new_ask_df['OrderQty']>0]
        new_bid_df=new_bid_df[new_bid_df['OrderQty']>0]
        return new_ask_df,new_bid_df
    def _trade(self,ask_df,bid_df):
        ask_series=ask_df.values
        bid_series=bid_df.values
        def match_once(ask_df,bid_df):
            last_price=self.stock_price[-1]
            if (ask_df.shape[0]==0) | (bid_df.shape[0]==0):
                return 0,0,ask_df,bid_df
            if ask_df[0][2]>bid_df[0][2]:
                #print("ask df:\n{}\n bid df:\n{}\n********************".format(ask_df,bid_df))
                return 0,0,ask_df,bid_df
            else:
                match_price=np.median([ask_df[0][2],bid_df[0][2],last_price])
                #print("ask df:\n{}\n bid df:\n{}\n********************".format(ask_df,bid_df))
                buyer=bid_df[0][0]
                seller=ask_df[0][0]
                buyerObj=self.agents[int(bid_df[0][0])]
                sellerObj=self.agents[int(ask_df[0][0])]
                volume=min(ask_df[0][3],bid_df[0][3])
                self.step_volume+=volume
                self.trade_count+=1
                df_temp_record={'time': [self.step_count],'price': [match_price],'buyer':[buyer],'seller':[seller],'qty':[volume]}
                #self.records=pd.concat([self.records,pd.DataFrame(df_temp_record)],ignore_index = True, axis = 0) #不再在每步记录此前所有的交易记录
            
                df_temp_buyer={'time': [self.step_count],'price': [match_price],'direction':[1],'qty':[volume]}
                df_temp_seller={'time': [self.step_count],'price': [match_price],'direction':[-1],'qty':[volume]}
                #self.schedule.agents[buyer].transaction_records=pd.concat([self.schedule.agents[buyer].transaction_records,pd.DataFrame(df_temp_buyer)],ignore_index = True, axis = 0)
                #self.schedule.agents[seller].transaction_records=pd.concat([self.schedule.agents[seller].transaction_records,pd.DataFrame(df_temp_seller)],ignore_index = True, axis = 0)
                buyerObj['buy_trades']+=volume
                sellerObj['sell_trades']+=volume
                
                if buyerObj['trader_type']==0:
                    buyerObj['fee']+=volume*match_price*0.0003
                else:
                    buyerObj['fee']+=volume*match_price*0.0001
                if sellerObj['trader_type']==0:
                    sellerObj['fee']+=volume*match_price*0.0003
                else:
                    sellerObj['fee']+=volume*match_price*0.0001
                buyerObj['position'] += volume
                if buyerObj['position']!=0:
                    buyerObj['weighted_cost']=(buyerObj['weighted_cost']*(buyerObj['position']-volume)+volume*match_price)/buyerObj['position']
                else:
                    buyerObj['close_profit']+=volume*(match_price-buyerObj['weighted_cost'])
                    buyerObj['weighted_cost']=0
                sellerObj['position'] -= volume
                if sellerObj['position']!=0:
                    sellerObj['weighted_cost']=(sellerObj['weighted_cost']*(sellerObj['position']+volume)-volume*match_price)/sellerObj['position']
                else:
                    sellerObj['close_profit']+=volume*(match_price-sellerObj['weighted_cost'])
                    sellerObj['weighted_cost']=0
                
                last_price=match_price
                if ask_df[0][3]> bid_df[0][3]:
                    
                    ask_df[0][3]=ask_df[0][3]-bid_df[0][3]
                    bid_df=bid_df[1:]
                elif ask_df[0][3]== bid_df[0][3]:
                    bid_df=bid_df[1:]
                    ask_df=ask_df[1:]
                else:
                    
                    bid_df[0][3]=bid_df[0][3]-ask_df[0][3]
                    ask_df=ask_df[1:]
                    
                if bid_df.shape[0]*ask_df.shape[0]==0:
                    return 0,match_price,ask_df,bid_df
                else:   
                    return 1,match_price,ask_df,bid_df
        if ask_series[0][2] is not None:
            res=1
            while res>0:
                res,match_price,ask_series,bid_series=match_once(ask_series,bid_series)
                if match_price>0:
                    self.stock_price.append(match_price)
        
        self.current_price=self.stock_price[-1]
        self.close_price.append(self.current_price)
        converted_ask_df=pd.DataFrame(ask_series,columns=["ID",'position','guess','OrderQty','direction','status','type','time'])
        converted_bid_df=pd.DataFrame(bid_series,columns=["ID",'position','guess','OrderQty','direction','status','type','time'])
        converted_ask_df=converted_ask_df[converted_ask_df['OrderQty']>0]
        converted_bid_df=converted_bid_df[converted_bid_df['OrderQty']>0]
        return converted_ask_df,converted_bid_df


