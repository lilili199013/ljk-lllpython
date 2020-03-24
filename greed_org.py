import random,copy,math,time

L_1=[47, 15, 29, 57, 52, 40, 20, 44, 45, 65, 12, 32, 50, 67, 33, 9, 30, 42, 11, 13, 60, 
18, 8, 59, 23, 31, 28, 6, 41, 55, 66, 19, 49, 24, 1, 56, 5, 43, 38, 39, 46, 70, 22, 26, 
4, 36, 25, 3, 69, 17, 61, 37, 7, 63, 21, 51, 54, 35, 2, 10, 34, 53, 62, 16, 27, 58, 48, 64, 68, 14]
L_2=[58021.555, 58188.973, 58438.129, 58198.211, 58139.844, 58130.465, 58224.496, 58210.168, 
58008.453, 58201.094, 58151.91, 58088.086, 58245.402, 58153.391, 58309.773, 57871.414, 58140.707, 
57811.164, 58446.582, 58526.051, 58034.395, 58022.969, 58467.906, 57427.84, 58113.273, 58020.02, 
58064.785, 58328.859, 58158.246, 58165.406, 57683.879, 58033.598, 58147.195, 58269.109, 58137.215, 
58146.219, 58490.207, 58285.211, 58373.438, 57637.551, 58130.953, 58334.555, 58334.188, 58211.605, 
58104.137, 57979.859, 58101.453, 57959.086, 58108.949, 58065.789, 58145.563, 58012.012, 58358.555, 
58045.586, 58227.848, 57842.711, 58148.449, 58487.664, 58519.691, 57871.137, 58167.398, 58023.844, 
57908.699, 57956.043, 58291.84, 58421.148, 57879.496, 57877.262, 57843.551, 58268.633]
L=zip(L_2,L_1)
L=dict(L)

def create_greed(num_geed,meta_list): #初始化序列数量
    gather_list=[]
    for i in range(num_geed):
        random.shuffle(meta_list)
        tem=copy.deepcopy(meta_list)
        gather_list.extend([tem])
    return gather_list

def calculate(list_list):#计算每个序列的值
    num_vane=len(list_list[0])
    sita=[i*((2*math.pi)/num_vane) for i in range(num_vane)]
    def map_list(each_list):
        M_x,M_y=0,0
        for i in range(num_vane):
            M_x+=math.sin(sita[i])*each_list[i]
            M_y+=math.cos(sita[i])*each_list[i]
        return (M_x**2+M_y**2)**(1/2)
    goal_list=list(map(map_list,list_list))
    return goal_list


def sort_greed(num_interation,list_list):#每个序列进行排序
    num_vane=len(list_list[0])
    sita=[i*((2*math.pi)/num_vane) for i in range(num_vane)]
    def sort_map(each_list):#用于映射
        def calculate_inner(each_list):#计算序列的值（目标函数）
            M_x,M_y=0,0
            for i in range(num_vane):
                M_x+=(math.sin(sita[i]))*each_list[i]
                M_y+=(math.cos(sita[i]))*each_list[i]
            return (M_x**2+M_y**2)**(1/2)
        
        for iters in range(num_interation):
            for times in range(20):
                rnd_1=random.randint(0,num_vane-1)
                rnd_2=random.randint(0,num_vane-1)
                while rnd_1==rnd_2:
                    rnd_2=random.randint(0,num_vane-1)
                each_list_new=copy.deepcopy(each_list)
                each_list_new[rnd_1],each_list_new[rnd_2]=each_list[rnd_2],each_list[rnd_1]
                out_old,out_new=calculate_inner(each_list),calculate_inner(each_list_new)
            if out_new<out_old:
                each_list=copy.deepcopy(each_list_new)
        return each_list
    out_list=list(map(sort_map,list_list))
    return out_list

num_times,num_lists=2000,20
time_start=time.time()
out_list=create_greed(num_lists,L_2)
out_list=sort_greed(num_times,out_list)
value_list=calculate(out_list)
time_end=time.time()
time_cost=time_end-time_start
print('运行时间为%s,%s个初始序列迭代%s次的结果:'%(time_cost,num_lists,num_times),value_list)


