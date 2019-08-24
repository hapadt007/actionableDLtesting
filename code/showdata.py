#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:30:19 2019

@author: qq
"""

import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import data_warehouse
import cliffsDelta
import scipy.stats as st

def averagenum(num):
    nsum = 0
    for i in range(len(num)):
        nsum += num[i]
    return 1.0*nsum / len(num)


def multiply_100(lst):
    tmp_lst=lst
    for i in range(len(lst)):
        for j in range(len(lst[i])):
            tmp_lst[i][j]=lst[i][j]*100.0
    return tmp_lst

def show_order_diffC():
    lengthlst=[100,300,500,700,900,1100,1300,1500,1700,2000,2500,3000]
    order_acclst=[[0.7623, 0.7602, 0.7637, 0.7625, 0.7625],#100
                  [0.7656, 0.7657, 0.7651, 0.7666, 0.7654],#300
                  [0.7691, 0.7707, 0.767, 0.7668, 0.7682],#500
                  [0.7757, 0.7708, 0.7739, 0.772, 0.7721],
                  [0.7739, 0.7738, 0.7765, 0.777, 0.7758],
                  [0.7787, 0.7769, 0.7782, 0.7768, 0.7765],
                  [0.7788, 0.7795, 0.7791, 0.7768, 0.7779],
                  [0.7768, 0.7841, 0.7816, 0.7783, 0.7821],
                  [0.786, 0.7879, 0.7815, 0.7843, 0.7842],
                  [0.788, 0.7882, 0.7873, 0.7851, 0.7871],
                  [0.7886, 0.7923, 0.7891, 0.7894, 0.7897],
                  [0.7913, 0.7898, 0.7889, 0.7891, 0.789]]
    random_acclst=[[0.7618, 0.7595, 0.7611, 0.7622, 0.7601],
                   [0.7615, 0.7642, 0.7633, 0.7632, 0.7613],
                   [0.7665, 0.7644, 0.7638, 0.7672, 0.7643],
                   [0.7658, 0.7653, 0.766, 0.767, 0.7656],
                   [0.7655, 0.7648, 0.7656, 0.7684, 0.7672],
                   [0.7656, 0.768, 0.769, 0.764, 0.767],
                   [0.7688, 0.7692, 0.7692, 0.768, 0.7697],
                   [0.7705, 0.7695, 0.7693, 0.7684, 0.768],
                   [0.7704, 0.772, 0.7691, 0.7689, 0.7674],
                   [0.769, 0.7715, 0.7703, 0.7701, 0.7714],
                   [0.7727, 0.7689, 0.7749, 0.7739, 0.7743],
                   [0.7747, 0.7733, 0.7772, 0.777, 0.7761]]
    x =lengthlst
    y_order_acc=[]
    y_random_acc=[]
    y_diff=[]
    for i in range(12):
        y_order_acc.append(averagenum(order_acclst[i]))
        y_random_acc.append(averagenum(random_acclst[i]))
        y_diff.append(averagenum(order_acclst[i])-averagenum(random_acclst[i]))
    '''
    plt.plot(x,y_order_acc,'blue')
    #plt.xlabel
    plt.plot(x,y_random_acc,'r')
    plt.ylabel('Retrain Accuracy')
    plt.xlabel('Sample Size')
    plt.legend(['order selected','random selected'])
    plt.show()
    '''
    print y_diff
    #y_diff=y_order_acc-y_random_acc
    plt.plot(x,y_diff,'r')
    plt.ylabel('Retrain Accuracy Improvement')
    plt.xlabel('Sample Size')
    #plt.legend(['order selected','random selected'])
    plt.show()
    return





def show_order_diff_A_125percent():
    dictA = data_warehouse.dictA
    lengthlst=[1,2,5,10]
    name='Rotate_down2'
    order_acclst=dictA[name+'order']
    random_acclst=dictA[name+'random']

    order_acclst= multiply_100(order_acclst)
    random_acclst= multiply_100(random_acclst)
    #print order_acclst
    
    x =lengthlst
    y_order_acc=[]
    y_random_acc=[]
    y_diff=[]
    for i in range(len(lengthlst)):
        avgorder=averagenum(order_acclst[i])
        y_order_acc.append(avgorder)
        origin=81.81
        avgrandom=averagenum(random_acclst[i])
        y_random_acc.append(avgrandom)
        print('%.4f&%.4f&%.4f'% (avgrandom, avgorder, (avgorder-origin)/(avgrandom-origin)))
        #print('average order_acc:%.4f'% averagenum(order_acclst[i]))
        y_diff.append(averagenum(order_acclst[i])-averagenum(random_acclst[i]))
        
        #print('E-value:%.4f'% ((averagenum(order_acclst[i])-origin)/(averagenum(random_acclst[i])-origin)))
    
    #print y_diff
    plt.plot(x,y_order_acc,'blue')
    #plt.xlabel
    
    plt.plot(x,y_random_acc,'r')
    plt.ylabel('Retrain Accuracy(%)')
    plt.xlabel('Sample Ratio(%)')
    plt.title("Details of Retrain in Data "+name)
    plt.legend(['order selected','random selected'])
    plt.savefig("../figure/125A/"+name+"a.png",dpi=300)
    plt.show()
    
    
    #print y_diff
    #y_diff=y_order_acc-y_random_acc
    plt.plot(x,y_diff,'r')
    plt.title("Retrain Accuracy Improvement in Data "+name)
    plt.ylabel('Accuracy Improvement(%)')
    plt.xlabel('Sample Ratio(%)')
    #plt.legend(['order selected','random selected'])
    plt.savefig("../figure/125A/"+name+"b.png",dpi=300)
    plt.show()
    
    xlabels=['1','2','5','10']
    
    plt.boxplot(order_acclst,widths=0.2,positions=[1+0.1,2+0.1,3.1,4.1],labels=xlabels,boxprops = {'color':'magenta'})
    offset=0.3
    plt.boxplot(random_acclst,widths=0.2,positions=[1+offset,2+offset,3+offset,4+offset],labels=xlabels,boxprops = {'color':'black'})
    plt.legend(labels=['order selected','random selected'])
    plt.ylabel('Retrain Accuracy(%)')
    plt.xlabel('Sample Ratio(%)')
    plt.title("Details of Retrain in Data "+name)
    plt.savefig("../figure/125A/"+name+"c.png",dpi=300)
    plt.show()
    return


def show_improve_random_A():
    dictA = data_warehouse.dictA_Discuss
    #lengthlst=[1,2,5,10]
    lengthlst=[5,10,20,30]
    y_avgimprove_lst=[[0 for i in range(8)]for j in range(4)] 
    y_order_avgacc_lst=[[] for j in range(len(lengthlst))]#四个子list
    y_random_avgacc_lst=[[] for j in range(len(lengthlst))]
    #y_random_wrong_avgacc_lst=[[] for j in range(len(lengthlst))]
    #y_order_wrong_avgacc_lst=[[] for j in range(len(lengthlst))]
    namelst=['Rotate_down2','Rotate_up2','Down2','Down3','Left3','Right3','Rotate_left2','Up3']
    
    for name in namelst:
        order_acclst=dictA[name+'order']
        random_acclst=dictA[name+'random']
        #random_wrong_acclst=dictA[name+'randomwrong']
        #order_wrong_acclst=dictA[name+'orderwrong']
        #变成百分比形式，所以每个都乘以100
        order_acclst= multiply_100(order_acclst) 
        random_acclst= multiply_100(random_acclst) 
        #random_wrong_acclst= multiply_100(random_wrong_acclst) 
        #order_wrong_acclst= multiply_100(order_wrong_acclst) 
        for i in range(len(lengthlst)):
                y_order_avgacc_lst[i].append(averagenum(order_acclst[i]))
                y_random_avgacc_lst[i].append(averagenum(random_acclst[i]))
         #       y_random_wrong_avgacc_lst[i].append(averagenum(random_wrong_acclst[i]))
          #      y_order_wrong_avgacc_lst[i].append(averagenum(order_wrong_acclst[i]))
    for i in range(4):
        for j in range(8):
            y_avgimprove_lst[i][j]=y_order_avgacc_lst[i][j]-y_random_avgacc_lst[i][j]
    bias=0.5
    first=0
    #xlabels=['1','2','5','10']
    xlabels=['5','10','20','30']
    
    plt.boxplot(y_avgimprove_lst,widths=0.4,positions=[first,first+bias,first+bias*2,first+bias*3],labels=xlabels,boxprops = {'color':'magenta'})
    plt.yticks(fontsize=15.0)
    plt.xticks(fontsize=15.0)   
    plt.ylabel('Improved Retrain Accuracy(%)',{'size':18})
    plt.xlabel('Sample Ratio(%)',{'size':18})
    #plt.title("Details of Improvement in ModelA (Rank+Entire VS Random+Entire)")
    plt.savefig("../figure/A/improve_random_entireA_discuss.png",dpi=300)
    plt.show()
    
    '''
    for i in range(4):
        for j in range(8):
            y_avgimprove_lst[i][j]=y_order_avgacc_lst[i][j]-y_random_wrong_avgacc_lst[i][j]    
    
    plt.boxplot(y_avgimprove_lst,widths=0.4,positions=[first,first+bias,first+bias*2,first+bias*3],labels=xlabels,boxprops = {'color':'magenta'})
    plt.yticks(fontsize=15.0)
    plt.xticks(fontsize=15.0)  
    plt.ylabel('Improved Retrain Accuracy(%)',{'size':18})
    plt.xlabel('Sample Ratio(%)',{'size':18})
    #plt.title("Details of Improvement in ModelA (Rank+Entire VS Random+Fail)")
    plt.savefig("../figure/A/improve_random_failA.png",dpi=300)
    plt.show()
    
    
    for i in range(4):
        for j in range(8):
            y_avgimprove_lst[i][j]=y_order_avgacc_lst[i][j]-y_order_wrong_avgacc_lst[i][j]    
    bias=0.5
    first=0
    plt.boxplot(y_avgimprove_lst,widths=0.4,positions=[first,first+bias,first+bias*2,first+bias*3],labels=xlabels,boxprops = {'color':'magenta'})
    plt.yticks(fontsize=15.0)
    plt.xticks(fontsize=15.0)   
    plt.ylabel('Improved Retrain Accuracy(%)',{'size':18})
    plt.xlabel('Sample Ratio(%)',{'size':18})
    #plt.title("Details of Improvement in ModelA (Rank+Entire VS Rank+Fail)")
    plt.savefig("../figure/A/improve_rank_failA.png",dpi=300)
    plt.show()
    '''
    return


def show_improve_random_B():
    dictB = data_warehouse.dictB_Discuss
    #lengthlst=[1,2,5,10]
    lengthlst=[5,10,20,30]
    y_avgimprove_lst=[[0 for i in range(6)]for j in range(4)] 
    y_order_avgacc_lst=[[] for j in range(len(lengthlst))]#四个子list
    y_random_avgacc_lst=[[] for j in range(len(lengthlst))]
    y_random_wrong_avgacc_lst=[[] for j in range(len(lengthlst))]
    y_order_wrong_avgacc_lst=[[] for j in range(len(lengthlst))]
    namelst=['Rotate_down2','Rotate_up2','Down3','Left3','Right3','Up3']
    
    for name in namelst:
        order_acclst=dictB[name+'order']
        random_acclst=dictB[name+'random']
        #random_wrong_acclst=dictB[name+'randomwrong']
        #order_wrong_acclst=dictB[name+'orderwrong']
        #变成百分比形式，所以每个都乘以100
        order_acclst= multiply_100(order_acclst) 
        random_acclst= multiply_100(random_acclst) 
        #random_wrong_acclst= multiply_100(random_wrong_acclst) 
        #order_wrong_acclst= multiply_100(order_wrong_acclst) 
        for i in range(len(lengthlst)):
                y_order_avgacc_lst[i].append(averagenum(order_acclst[i]))
                y_random_avgacc_lst[i].append(averagenum(random_acclst[i]))
                #y_random_wrong_avgacc_lst[i].append(averagenum(random_wrong_acclst[i]))
                #y_order_wrong_avgacc_lst[i].append(averagenum(order_wrong_acclst[i]))
    for i in range(4):
        for j in range(6):
            y_avgimprove_lst[i][j]=y_order_avgacc_lst[i][j]-y_random_avgacc_lst[i][j]
    
    #xlabels=['1','2','5','10']
    xlabels=['5','10','20','30']
    bias=0.5
    first=0
    plt.boxplot(y_avgimprove_lst,widths=0.4,positions=[first,first+bias,first+bias*2,first+bias*3],labels=xlabels,boxprops = {'color':'magenta'})
    plt.yticks(fontsize=15.0)
    plt.xticks(fontsize=15.0)   
    plt.ylabel('Improved Retrain Accuracy(%)',{'size':18})
    plt.xlabel('Sample Ratio(%)',{'size':18})
    #plt.title("Details of Improvement in ModelB (Rank+Entire VS Random+Entire)")
    plt.savefig("../figure/B/improve_random_entireB_discuss.png")
    plt.show()
    
    '''
    for i in range(4):
        for j in range(6):
            y_avgimprove_lst[i][j]=y_order_avgacc_lst[i][j]-y_random_wrong_avgacc_lst[i][j]    
    
    plt.boxplot(y_avgimprove_lst,widths=0.4,positions=[first,first+bias,first+bias*2,first+bias*3],labels=xlabels,boxprops = {'color':'magenta'})
    plt.yticks(fontsize=15.0)
    plt.xticks(fontsize=15.0)   
    plt.ylabel('Improved Retrain Accuracy(%)',{'size':18})
    plt.xlabel('Sample Ratio(%)',{'size':18})
    #plt.title("Details of Improvement in ModelB (Rank+Entire VS Random+Fail)")
    plt.savefig("../figure/B/improve_random_failB.png",dpi=300)
    plt.show()
    
    
    for i in range(4):
        for j in range(6):
            y_avgimprove_lst[i][j]=y_order_avgacc_lst[i][j]-y_order_wrong_avgacc_lst[i][j]    
    
    plt.boxplot(y_avgimprove_lst,widths=0.4,positions=[first,first+bias,first+bias*2,first+bias*3],labels=xlabels,boxprops = {'color':'magenta'})
    plt.yticks(fontsize=15.0)
    plt.xticks(fontsize=15.0)
    plt.ylabel('Improved Retrain Accuracy(%)',{'size':18})
    plt.xlabel('Sample Ratio(%)',{'size':18})
    #plt.title("Details of Improvement in ModelB (Rank+Entire VS Rank+Fail)")
    plt.savefig("../figure/B/improve_rank_failB.png",dpi=300)
    plt.show()
    '''
    return


def show_order_diff_B_125percent():
    dictB =data_warehouse.dictB

    #print("swj")
    lengthlst=[1,2,5,10]
    #lengthlst=[5,10,20,30]
    name='Up3'
    order_acclst=dictB[name+'order']
    random_acclst=dictB[name+'random']

    order_acclst= multiply_100(order_acclst)
    random_acclst= multiply_100(random_acclst)
    #print order_acclst
    
    x =lengthlst
    y_order_acc=[]
    y_random_acc=[]
    y_diff=[]
    for i in range(len(lengthlst)):
        avgorder=averagenum(order_acclst[i])
        y_order_acc.append(avgorder)
        origin=75.38
        avgrandom=averagenum(random_acclst[i])
        y_random_acc.append(avgrandom)
        print('%.4f&%.4f&%.4f'% (avgrandom, avgorder, (avgorder-origin)/(avgrandom-origin)))
        #print('average order_acc:%.4f'% averagenum(order_acclst[i]))
        y_diff.append(averagenum(order_acclst[i])-averagenum(random_acclst[i]))
    
    #print y_diff
    plt.plot(x,y_order_acc,'blue')
    #plt.xlabel
    
    plt.plot(x,y_random_acc,'r')
    plt.ylabel('Retrain Accuracy(%)')
    plt.xlabel('Sample Ratio(%)')
    plt.title("Details of Retrain in Data "+name)
    plt.legend(['order selected','random selected'])
    plt.savefig("../figure/125B/"+name+"a.png",dpi=300)
    plt.show()
    
    
    #print y_diff
    #y_diff=y_order_acc-y_random_acc
    plt.plot(x,y_diff,'r')
    plt.title("Retrain Accuracy Improvement in Data "+name)
    plt.ylabel('Accuracy Improvement(%)')
    plt.xlabel('Sample Ratio(%)')
    #plt.legend(['order selected','random selected'])
    plt.savefig("../figure/125B/"+name+"b.png",dpi=300)
    plt.show()
    
    xlabels=['1','2','5','10']
    
    plt.boxplot(order_acclst,widths=0.2,positions=[1+0.1,2+0.1,3.1,4.1],labels=xlabels,boxprops = {'color':'magenta'})
    offset=0.3
    plt.boxplot(random_acclst,widths=0.2,positions=[1+offset,2+offset,3+offset,4+offset],labels=xlabels,boxprops = {'color':'black'})
    plt.legend(labels=['order selected','random selected'])
    plt.ylabel('Retrain Accuracy(%)')
    plt.xlabel('Sample Ratio(%)')
    plt.title("Details of Retrain in Data "+name)
    plt.savefig("../figure/125B/"+name+"c.png",dpi=300)
    plt.show()
    return

    
def A_into_table():
    dictA = data_warehouse.dictA
    lengthlst=[1,2,5,10]
    #namelst=['Rotate_down2','Rotate_up2','Down2','Down3','Left3','Right3','Rotate_left2','Up3']
    origin_dct={'Rotate_down2':81.81,'Rotate_up2':84.98,'Down2':84.78,'Down3':61.91,'Left3':72.42,'Right3':82.09,'Rotate_left2':84.04,'Up3':68.59}
    
    name='Up3'
    order_acclst=dictA[name+'order']
    #randomwrong
    #orderwrong
    random_acclst=dictA[name+'orderwrong']

    order_acclst= multiply_100(order_acclst)
    random_acclst= multiply_100(random_acclst)
    #print order_acclst
    #namelst=['Rotate_down2','Rotate_up2','Down2','Down3',
    #'Left3','Right3','Rotate_left2','Up3']
    
    #x =lengthlst
    y_order_acc=[]
    y_random_acc=[]
    y_diff=[]
    for i in range(len(lengthlst)):
        avgorder=averagenum(order_acclst[i])
        y_order_acc.append(avgorder)
        origin=origin_dct[name]
        avgrandom=averagenum(random_acclst[i])
        d, res = cliffsDelta.cliffsDelta(order_acclst[i],random_acclst[i])
        sta,p_value =st.ranksums(random_acclst[i], order_acclst[i])
        y_random_acc.append(avgrandom)
        print('%.2f&%.2f(%.2f&%.4f)&%.2f'% (avgrandom, avgorder,d,p_value, (avgorder-origin)/(avgrandom-origin)))
        #print('average order_acc:%.4f'% averagenum(order_acclst[i]))
        y_diff.append(averagenum(order_acclst[i])-averagenum(random_acclst[i]))
    return

#A_into_table_discuss
def A_into_table_discuss():
    dictA = data_warehouse.dictA
    lengthlst=[1,2,5,10]
    #namelst=['Rotate_down2','Rotate_up2','Down2','Down3','Left3','Right3','Rotate_left2','Up3']
    origin_dct={'Rotate_down2':81.81,'Rotate_up2':84.98,'Down2':84.78,'Down3':61.91,
                'Left3':72.42,'Right3':82.09,'Rotate_left2':84.04,'Up3':68.59}
    
    name='Up3'
    order_acclst=dictA[name+'order']
    #randomwrong
    #orderwrong
    random_acclst=dictA[name+'random']

    order_acclst= multiply_100(order_acclst)
    random_acclst= multiply_100(random_acclst)
    #print order_acclst
    #namelst=['Rotate_down2','Rotate_up2','Down2','Down3',
    #'Left3','Right3','Rotate_left2','Up3']
    
    #x =lengthlst
    y_order_acc=[]
    y_random_acc=[]
    y_diff=[]
    
    avgorder=averagenum(order_acclst[1])
    y_order_acc.append(avgorder)
    origin=origin_dct[name]
    avgrandom=averagenum(random_acclst[2])
    d, res = cliffsDelta.cliffsDelta(order_acclst[1],random_acclst[2])
    sta,p_value =st.ranksums(random_acclst[2], order_acclst[1])
    y_random_acc.append(avgrandom)
    print('%.2f&%.2f(%.2f&%.4f)&%.2f'% (avgrandom, avgorder,d,p_value, (avgorder-origin)/(avgrandom-origin)))
    #print('average order_acc:%.4f'% averagenum(order_acclst[i]))
    #y_diff.append(averagenum(order_acclst[1])-averagenum(random_acclst[2]))
    return

def B_into_table_discuss():
    dictA = data_warehouse.dictB
    lengthlst=[1,2,5,10]
    #namelst=['Rotate_down2','Rotate_up2','Down2','Down3','Left3','Right3','Rotate_left2','Up3']
    origin_dct={'Rotate_down2':88.28,'Rotate_up2':88.84,
                'Down3':71.54,'Left3':83.7,'Right3':83.15,'Up3':75.38}
    name='Up3'
    order_acclst=dictA[name+'order']
    #randomwrong
    #orderwrong
    random_acclst=dictA[name+'random']

    order_acclst= multiply_100(order_acclst)
    random_acclst= multiply_100(random_acclst)
    #print order_acclst
    #namelst=['Rotate_down2','Rotate_up2','Down2','Down3',
    #'Left3','Right3','Rotate_left2','Up3']
    
    #x =lengthlst
    y_order_acc=[]
    y_random_acc=[]
    y_diff=[]
    
    avgorder=averagenum(order_acclst[1])
    y_order_acc.append(avgorder)
    origin=origin_dct[name]
    avgrandom=averagenum(random_acclst[2])
    d, res = cliffsDelta.cliffsDelta(order_acclst[1],random_acclst[2])
    sta,p_value =st.ranksums(random_acclst[2], order_acclst[1])
    y_random_acc.append(avgrandom)
    print('%.2f&%.2f(%.2f&%.4f)&%.2f'% (avgrandom, avgorder,d,p_value, (avgorder-origin)/(avgrandom-origin)))
    #print('average order_acc:%.4f'% averagenum(order_acclst[i]))
    #y_diff.append(averagenum(order_acclst[1])-averagenum(random_acclst[2]))
    return

def B_into_table():
    dictB = data_warehouse.dictB
    lengthlst=[1,2,5,10]
    #namelst=['Rotate_down2','Rotate_up2','Down3','Left3','Right3','Up3']
    name='Up3'
    order_acclst=dictB[name+'order']
    #orderwrong
    random_acclst=dictB[name+'randomwrong']

    order_acclst= multiply_100(order_acclst)
    random_acclst= multiply_100(random_acclst)
    #print order_acclst
    #namelst=['Rotate_down2','Rotate_up2','Down2','Down3',
    #'Left3','Right3','Rotate_left2','Up3']
    origin_dct={'Rotate_down2':88.28,'Rotate_up2':88.84,'Down3':71.54,'Left3':83.7,'Right3':83.15,'Up3':75.38}
    #x =lengthlst
    y_order_acc=[]
    y_random_acc=[]
    y_diff=[]
    for i in range(len(lengthlst)):
        avgorder=averagenum(order_acclst[i])
        y_order_acc.append(avgorder)
        origin=origin_dct[name]
        avgrandom=averagenum(random_acclst[i])
        d, res = cliffsDelta.cliffsDelta(order_acclst[i],random_acclst[i])
        y_random_acc.append(avgrandom)
        sta,p_value =st.ranksums(random_acclst[i], order_acclst[i])
        print('%.2f&%.2f(%.2f&%.4f)&%.2f'% (avgrandom, avgorder,d, p_value,(avgorder-origin)/(avgrandom-origin)))
        #print('average order_acc:%.4f'% averagenum(order_acclst[i]))
        y_diff.append(averagenum(order_acclst[i])-averagenum(random_acclst[i]))
    return

if __name__ == "__main__":
    #show_order_diff()
    B_into_table_discuss()