import numpy as np

def funcion_objetivo(x, problema_id=0):
    if problema_id == 0:
        sum1 = 5 * np.sum(x[0:4])
        sum2 = 5 * np.sum(np.power(x[0:4], 2))
        sum3 = np.sum(x[4:13])
        return sum1 - sum2 - sum3

    elif problema_id == 1:
        n = len(x)
        sum_cos4 = np.sum(np.cos(x)**4)
        prod_cos2 = np.prod(np.cos(x)**2)
        numerador = sum_cos4 - 2 * prod_cos2
        indices = np.arange(1, n+1)
        denominador = np.sqrt(np.sum(indices * x**2))
        
        return -np.abs(numerador / denominador)


    elif problema_id == 2:
        n = len(x)
        term1 = np.power(np.sqrt(n), n)
        term2 = np.prod(x)
        return -term1 * term2
    
    elif problema_id == 3:        
        term1 = 5.3578547 * x[2]**2
        term2 = 0.8356891 * x[0] * x[4]
        term3 = 37.293239 * x[0]
        constant = -40792.141
        return term1 + term2 + term3 + constant
    
    elif problema_id == 4:
        term1 = 3 * x[0]
        term2 = 0.000001 * x[0]**3
        term3 = 2 * x[1]
        term4 = (0.000002 / 3.0) * x[1]**3
        return term1 + term2 + term3 + term4    

    elif problema_id == 5:

        term1 = (x[0] - 10)**3
        term2 = (x[1] - 20)**3
        return term1 + term2

    elif problema_id == 6:

        term1 = x[0]**2 + x[1]**2 + x[0]*x[1] - 14*x[0] - 16*x[1]
        term2 = (x[2] - 10)**2 + 4*(x[3] - 5)**2 + (x[4] - 3)**2
        term3 = 2*(x[5] - 1)**2 + 5*x[6]**2 + 7*(x[7] - 11)**2
        term4 = 2*(x[8] - 10)**2 + (x[9] - 7)**2 + 45
        return term1 + term2 + term3 + term4

    elif problema_id == 7:

        numerador = -(np.sin(2 * np.pi * x[0])**3) * np.sin(2 * np.pi * x[1])
        denominador = (x[0]**3) * (x[0] + x[1])
             
        return numerador / denominador

    elif problema_id == 8:
        term1 = (x[0] - 10)**2 + 5*(x[1] - 12)**2 + x[2]**4 + 3*(x[3] - 11)**2
        term2 = 10*x[4]**6 + 7*x[5]**2 + x[6]**4 - 4*x[5]*x[6] - 10*x[5] - 8*x[6]
        return term1 + term2

    elif problema_id == 9:
        return x[0] + x[1] + x[2]

    elif problema_id == 10:
        return x[0]**2 + (x[1] - 1)**2

    elif problema_id == 11:
        term1 = (x[0] - 5)**2
        term2 = (x[1] - 5)**2
        term3 = (x[2] - 5)**2
        return -(100 - term1 - term2 - term3) / 100.0

    elif problema_id == 12:
        return np.exp(x[0] * x[1] * x[2] * x[3] * x[4])

    elif problema_id == 13:

        c = np.array([-6.089, -17.164, -34.054, -5.914, -24.721, 
                      -14.986, -24.1, -10.708, -26.662, -22.179])

        if np.any(x <= 0) or np.sum(x) == 0:
            return np.inf

        suma_x = np.sum(x)
        log_term = np.log(x / suma_x)
        return np.sum(x * (c + log_term))

    elif problema_id == 14:
        return 1000 - x[0]**2 - 2*x[1]**2 - x[2]**2 - x[0]*x[1] - x[0]*x[2]

    elif problema_id == 15:
            # Primero calculamos todas las variables auxiliares
            y1 = x[1] + x[2] + 41.6
            c1 = 0.024*x[3] - 4.62
            y2 = (12.5/c1) + 12
            c2 = 0.0003535*(x[0]**2) + (0.5311*x[0]) + (0.08705*y2*x[0])
            c3 = (0.052*x[0]) + 78 + (0.002377*y2*x[0])
            y3 = c2/c3
            y4 = 19*y3
            c4 = (0.04782*(x[0] - y3)) + (0.1956*((x[0] - y3))**2)/x[1] + (0.6376*y4) + (1.594*y3)
            c5 = 100*x[1]
            c6 = x[0] - y3 - y4
            c7 = 0.950 - (c4/c5)
            y5 = c6*c7
            y6 = x[0] - y5 - y4 - y3
            c8 = (y5 + y4)*0.995
            y7 = c8/y1
            y8 = c8/3798
            c9 = y7 - ((0.0663*y7)/y8) - 0.3153
            y9 = (96.82/c9) + (0.321*y1)
            y10 = (1.29*y5) + (1.258*y4) + (2.29*y3) + (1.71*y6)
            y11 = (1.71*x[0]) - (0.452*y4) + (0.580*y3)
            c10 = 12.3/752.3
            c11 = (1.75*y2)*(0.995*x[0])
            c12 = (0.995*y10) + 1998
            y12 = (c10*x[0]) + (c11/c12)
            y13 = c12 - (1.75*y2)
            y14 = 3623 + (64.4*x[1]) + (58.4*x[2]) + (146312/(y9 + x[4]))
            c13 = (0.995*y10) + (60.8*x[1]) + (48*x[3]) - (0.1121*y14) - 5095
            y15 = y13/c13
            y16 = 148000 - (331000*y15) + (40*y13) - (61*y15*y13)
            c14 = (2324*y10) - (28740000*y2)
            y17 = 14130000 - (1328*y10) - (531*y11) + (c14/c12)
            c15 = (y13/y15) - (y13/0.52)
            c16 = 1.104 - (0.72*y15)
            c17 = y9 + x[4]
            
            # Función objetivo g16
            f = (0.000117 * y14) + 0.1365 + (0.00002358 * y13) + (0.000001502 * y16) + (0.0321 * y12) + (0.004324 * y5) + (0.0001 * (c15 / c16)) + (37.48 * (y2 / c12)) - (0.0000005843 * y17)

            return f

    elif problema_id == 16:

        if 0 <= x[0] < 300:
            f1 = 30 * x[0]
        else:  
            f1 = 31 * x[0]
        
        if 0 <= x[1] < 100:
            f2 = 28 * x[1]
        elif 100 <= x[1] < 200:
            f2 = 29 * x[1]
        else:  # 200 <= x1 < 1000
            f2 = 30 * x[1]
        
        # La función objetivo es la suma de f(x1) y f(x2)
        return f1 + f2

    elif problema_id == 17:
        term1 = x[0]*x[3] - x[1]*x[2] + x[2]*x[8] 
        term2 = - x[4]*x[8] + x[4]*x[7] - x[5]*x[6]
        return -0.5 * (term1 + term2)

    elif problema_id == 18:
        c = [
            [-15, -27, -36, -18, -12],  # e_j
            [30, -20, -10, 32, -10],    # c_1j
            [-20, 39, -6, -31, 32],     # c_2j
            [-10, -6, 10, -6, -10],     # c_3j
            [32, -31, -6, 39, -20],     # c_4j
            [-10, 32, -10, -20, 30],    # c_5j
        ]
        d = [4, 8, 10, 6, 2]
        b = [-40, -2, -0.25, -4, -4, -1, -40, -60, 5, 1]
        

        f = 0
        #Doble sumatoria (c)
        for j in range(5):
            for i in range(5):
                f += c[i+1][j] * x[10+i] * x[10+j]
        
        # Sumatoria (d)
        for j in range(5):
            f += 2 * d[j] * x[10+j]**3
        
        # Sumatoria (b)
        for i in range(10):
            f -= b[i] * x[i]
        
        return f

    elif problema_id == 19:

        a = [0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09,
            0.0693, 0.0577, 0.05, 0.2, 0.26, 0.55, 0.06, 0.1, 0.12, 0.18, 0.1, 0.09]
        f = 0
        for i in range(24):
            f += a[i] * x[i]
        
        return f

    elif problema_id == 20:
        return x[0]

    elif problema_id == 21:
        return x[0]

    elif problema_id == 22:
        return -9.0*x[4] - 15.0*x[7] + 6.0*x[0] + 16.0*x[1] + 10.0*(x[5] + x[6])

    elif problema_id == 23:
        return -x[0] - x[1]
    
    else:
        raise ValueError(f"Función objetivo para problema ID {problema_id} no implementada")


def restricciones(x, problema_id=0):
    if problema_id == 0:
        g = np.zeros(9)
        g[0] = 2*x[0] + 2*x[1] + x[9] + x[10] - 10
        g[1] = 2*x[0] + 2*x[2] + x[9] + x[11] - 10
        g[2] = 2*x[1] + 2*x[2] + x[10] + x[11] - 10
        g[3] = -8*x[0] + x[9]
        g[4] = -8*x[1] + x[10]
        g[5] = -8*x[2] + x[11]
        g[6] = -2*x[3] - x[4] + x[9]
        g[7] = -2*x[5] - x[6] + x[10]
        g[8] = -2*x[7] - x[8] + x[11]
        return g
    
    elif problema_id == 1:

        g = np.zeros(2)

        g[0] = 0.75 - np.prod(x)
        g[1] = np.sum(x) - 7.5 * len(x)
        
        return g

    elif problema_id == 2:
         
         g = np.zeros(1)
         h1 = np.sum(np.power(x, 2)) - 1
         g[0]= np.abs(h1)
         return g

    elif problema_id == 3:
        
         g = np.zeros(6)
         
         g[0] = 85.334407 + 0.0056858*x[1]*x[4] + 0.0006262*x[0]*x[3] - 0.0022053*x[2]*x[4] - 92
         g[1] = -85.334407 - 0.0056858*x[1]*x[4] - 0.0006262*x[0]*x[3] + 0.0022053*x[2]*x[4] 
         g[2] = 80.51249 + 0.0071317*x[1]*x[4] + 0.0029955*x[0]*x[1] + 0.0021813*x[2]**2 - 110         
         g[3] = -80.51249 - 0.0071317*x[1]*x[4] - 0.0029955*x[0]*x[1] + 0.0021813*x[2]**2 + 90         
         g[4] = 9.300961 + 0.0047026*x[2]*x[4] + 0.0012547*x[0]*x[2] + 0.0019085*x[2]*x[3] - 25         
         g[5] = -9.300961 - 0.0047026*x[2]*x[4] - 0.0012547*x[0]*x[2] - 0.0019085*x[2]*x[3] + 20
         
         return g
    
    elif problema_id == 4:

        g = np.zeros(5)
        g[0] = -x[3] + x[2] - 0.55
        g[1] = -x[2] + x[3] - 0.55
        h3 = 1000 * np.sin(-x[2] - 0.25) + 1000 * np.sin(-x[3] - 0.25) + 894.8 - x[0]
        g[2] = np.abs(h3)
        h4 = 1000 * np.sin(x[2] - 0.25) + 1000 * np.sin(x[2] - x[3] - 0.25) + 894.8 - x[1]
        g[3] = np.abs(h4)
        h5 = 1000 * np.sin(x[3] - 0.25) + 1000 * np.sin(x[3] - x[2] - 0.25) + 1294.8
        g[4] = np.abs(h5)

        return g
    
    elif problema_id == 5:
        g = np.zeros(2) 

        g[0] = -(x[0] - 5)**2 - (x[1] - 5)**2 + 100
        g[1] = (x[0] - 6)**2 + (x[1] - 5)**2 - 82.81

        return g        

    elif problema_id == 6:

        g = np.zeros(8) 

        g[0] = -105 + 4*x[0] + 5*x[1] - 3*x[6] + 9*x[7]
        g[1] = 10*x[0] - 8*x[1] - 17*x[6] + 2*x[7]
        g[2] = -8*x[0] + 2*x[1] + 5*x[8] - 2*x[9] - 12
        g[3] = 3*(x[0] - 2)**2 + 4*(x[1] - 3)**2 + 2*x[2]**2 - 7*x[3] - 120
        g[4] = 5*x[0]**2 + 8*x[1] + (x[2] - 6)**2 - 2*x[3] - 40
        g[5] = x[0]**2 + 2*(x[1] - 2)**2 - 2*x[0]*x[1] + 14*x[4] - 6*x[5]
        g[6] = 0.5*(x[0] - 8)**2 + 2*(x[1] - 4)**2 + 3*x[4]**2 - x[5] - 30
        g[7] = -3*x[0] + 6*x[1] + 12*(x[8] - 8)**2 - 7*x[9]
        
        return g

    elif problema_id == 7:
        g = np.zeros(2)
        g[0] = x[0]**2 - x[1] + 1
        g[1] = 1 - x[0] + (x[1] - 4)**2
        return g

    elif problema_id == 8:
        g = np.zeros(4)
        g[0] = -127 + 2*x[0]**2 + 3*x[1]**4 + x[2] + 4*x[3]**2 + 5*x[4]
        g[1] = -282 + 7*x[0] + 3*x[1] + 10*x[2]**2 + x[3] - x[4]
        g[2] = -196 + 23*x[0] + x[1]**2 + 6*x[5]**2 - 8*x[6]
        g[3] = 4*x[0]**2 + x[1]**2 - 3*x[0]*x[1] + 2*x[2]**2 + 5*x[5] - 11*x[6]
        return g

    elif problema_id == 9:
        g = np.zeros(6)
        g[0] = -1 + 0.0025 * (x[3] + x[5])
        g[1] = -1 + 0.0025 * (x[4] + x[6] - x[3])
        g[2] = -1 + 0.01 * (x[7] - x[4])
        g[3] = -x[0] * x[5] + 833.33252 * x[3] + 100 * x[0] - 83333.333
        g[4] = -x[1] * x[6] + (1250 * x[4]) + (x[1] * x[3]) - (1250 * x[3])
        g[5] = -x[2] * x[7] + 1250000 + x[2] * x[4] - 2500 * x[4]
        return g

    elif problema_id == 10:
        g = np.zeros(1)
        h = x[1] - x[0]**2
        g[0] = np.abs(h)
        return g

    elif problema_id == 11:
        g = np.ones(1)  # Inicializamos como restricción violada
        
        for p in range(1, 10):
            for q in range(1, 10):
                for r in range(1, 10):
                    if (x[0] - p)**2 + (x[1] - q)**2 + (x[2] - r)**2 <= 0.0625:
                        g[0] = 0.0  # Restricción satisfecha
                        return g    # Terminamos la búsqueda al encontrar una esfera válida
        
        return g

    elif problema_id == 12:
        g = np.zeros(3)
        h1 = x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 - 10
        h2 = x[1]*x[2] - 5*x[3]*x[4]
        h3 = x[0]**3 + x[1]**3 + 1
        g[0] = np.abs(h1)
        g[1] = np.abs(h2)
        g[2] = np.abs(h3)
        return g

    elif problema_id == 13:
        g = np.zeros(3)
        h1 = x[0] + 2*x[1] + 2*x[2] + x[5] + x[9] - 2
        h2 = x[3] + 2*x[4] + x[5] + x[6] - 1
        h3 = x[2] + x[6] + x[7] + 2*x[8] + x[9] - 1
        g[0] = np.abs(h1)
        g[1] = np.abs(h2)
        g[2] = np.abs(h3)
        return g

    elif problema_id == 14:
        g = np.zeros(2)
        h1 = x[0]**2 + x[1]**2 + x[2]**2 - 25
        h2 = 8*x[0] + 14*x[1] + 7*x[2] - 56
        g[0] = np.abs(h1)
        g[1] = np.abs(h2)
        return g

    elif problema_id == 15:  
        y1 = x[1] + x[2] + 41.6
        c1 = 0.024*x[3] - 4.62
        y2 = (12.5/c1) + 12
        c2 = 0.0003535*(x[0]**2) + (0.5311*x[0]) + (0.08705*y2*x[0])
        c3 = (0.052*x[0]) + 78 + (0.002377*y2*x[0])
        y3 = c2/c3
        y4 = 19*y3
        c4 = (0.04782*(x[0] - y3)) + (0.1956*((x[0] - y3))**2)/x[1] + (0.6376*y4) + (1.594*y3)
        c5 = 100*x[1]
        c6 = x[0] - y3 - y4
        c7 = 0.950 - (c4/c5)
        y5 = c6*c7
        y6 = x[0] - y5 - y4 - y3
        c8 = (y5 + y4)*0.995
        y7 = c8/y1
        y8 = c8/3798
        c9 = y7 - ((0.0663*y7)/y8) - 0.3153
        y9 = (96.82/c9) + (0.321*y1)
        y10 = (1.29*y5) + (1.258*y4) + (2.29*y3) + (1.71*y6)
        y11 = (1.71*x[0]) - (0.452*y4) + (0.580*y3)
        c10 = 12.3/752.3
        c11 = (1.75*y2)*(0.995*x[0])
        c12 = (0.995*y10) + 1998
        y12 = (c10*x[0]) + (c11/c12)
        y13 = c12 - (1.75*y2)
        y14 = 3623 + (64.4*x[1]) + (58.4*x[2]) + (146312/(y9 + x[4]))
        c13 = (0.995*y10) + (60.8*x[1]) + (48*x[3]) - (0.1121*y14) - 5095
        y15 = y13/c13
        y16 = 148000 - (331000*y15) + (40*y13) - (61*y15*y13)
        c14 = (2324*y10) - (28740000*y2)
        y17 = 14130000 - (1328*y10) - (531*y11) + (c14/c12)
        c15 = (y13/y15) - (y13/0.52)
        c16 = 1.104 - (0.72*y15)
        c17 = y9 + x[4]
 
        g = np.zeros(38)
        # Restricciones g1 a g38
        g[0] = ((0.28/0.72)*y5) - y4          
        g[1] = x[2] - 1.5*x[1]          
        g[2] = (3496*(y2/c12)) - 21          
        g[3] = 110.6 + y1 - (62212/c17 )         
        g[4] = 213.1 - y1          
        g[5] = y1 - 405.23          
        g[6] = 17.505 - y2          
        g[7] = y2 - 1053.6667         
        g[8] = 11.275 - y3          
        g[9] = y3 - 35.03  
        g[10] = 214.228 - y4  
        g[11] = y4 - 665.585  
        g[12] = 7.458 - y5  
        g[13] = y5 - 584.463  
        g[14] = 0.961 - y6  
        g[15] = y6 - 265.916  
        g[16] = 1.612 - y7  
        g[17] = y7 - 7.046  
        g[18] = 0.146 - y8  
        g[19] = y8 - 0.222  
        g[20] = 107.99 - y9  
        g[21] = y9 - 273.366  
        g[22] = 922.693 - y10  
        g[23] = y10 - 1286.105  
        g[24] = 926.832 - y11  
        g[25] = y11 - 1444.046  
        g[26] = 18.766 - y12  
        g[27] = y12 - 537.141  
        g[28] = 1072.163 - y13  
        g[29] = y13 - 3247.039  
        g[30] = 8961.448 - y14  
        g[31] = y14 - 26844.086  
        g[32] = 0.063 - y15  
        g[33] = y15 - 0.386  
        g[34] = 71084.33 - y16  
        g[35] = -140000 + y16  
        g[36] = 2802713 - y17  
        g[37] = y17 - 12146108  
        
        return g

    elif problema_id == 16:
        g = np.zeros(4)
        
        h0 = -x[0] + 300 - (((x[2]*x[3])/131.078) * np.cos(1.48477 - x[5])) + (((0.90798*(x[2]**2))/131.078) * np.cos(1.47588))
        g[0] = np.abs(h0)
        h1 = -x[1] - (((x[2]*x[3])/131.078) * np.cos(1.48477 + x[5])) + (((0.90798*(x[3]**2))/131.078) * np.cos(1.47588))
        g[1] = np.abs(h1)
        h2 = -x[4] - (((x[2]*x[3])/131.078) * np.sin(1.48477 + x[5])) + (((0.90798*(x[3]**2))/131.078) * np.sin(1.47588))
        g[2] = np.abs(h2)
        h3 = 200 - (((x[2]*x[3])/131.078) * np.sin(1.48477 - x[5])) + (((0.90798*(x[2]**2))/131.078) * np.sin(1.47588))
        g[3] = np.abs(h3)

        return g

    elif problema_id == 17:
        g = np.zeros(13)
        g[0] = x[2]**2 + x[3]**2 - 1.0
        g[1] = x[8]**2 - 1.0
        g[2] = x[4]**2 + x[5]**2 - 1.0
        g[3] = x[0]**2 + (x[1] - x[8])**2 - 1.0
        g[4] = (x[0] - x[4])**2 + (x[1] - x[5])**2 - 1.0
        g[5] = (x[0] - x[6])**2 + (x[1] - x[7])**2 - 1.0
        g[6] = (x[2] - x[4])**2 + (x[3] - x[5])**2 - 1.0
        g[7] = (x[2] - x[6])**2 + (x[3] - x[7])**2 - 1.0
        g[8] = x[6]**2 + (x[7] - x[8])**2 - 1.0
        g[9] = x[1]*x[2] - x[0]*x[3]
        g[10] = -x[2]*x[8]
        g[11] = x[4]*x[8]
        g[12] = x[5]*x[6] - x[4]*x[7]
        return g

    elif problema_id == 18:

        c = [
            [-15, -27, -36, -18, -12],  
            [30, -20, -10, 32, -10],    
            [-20, 39, -6, -31, 32],     
            [-10, -6, 10, -6, -10],     
            [32, -31, -6, 39, -20],     
            [-10, 32, -10, -20, 30],    
        ]
        d = [4, 8, 10, 6, 2]
        e = c[0]
        
        a = [
            [-16, 2, 0, 1, 0],          
            [0, -2, 0, 0.4, 2],         
            [-3.5, 0, 2, 0, 0],         
            [0, -2, 0, -4, -1],         
            [0, -9, -2, 1, -2.8],       
            [2, 0, -4, 0, 0],           
            [-1, -1, -1, -1, -1],       
            [-1, -2, -3, -2, -1],       
            [1, 2, 3, 4, 5],            
            [1, 1, 1, 1, 1],            
        ]
        
        g = np.zeros(5)
        
        for j in range(5):
            # Sumatoria (c)
            term1 = 0
            for i in range(5):
                term1 += c[i+1][j] * x[10+i]
            
            # Segundo término (dj)
            term2 = 3 * d[j] * x[10+j]**2
            
            # Tercero (ej)
            term3 = e[j]
            
            # Sumatoria (aij)
            term4 = 0
            for i in range(10):
                term4 += a[i][j] * x[i]
            
            g[j] = -2 * term1 - term2 - term3 + term4
        
        return g

    elif problema_id == 19:
        b = [44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 62.501, 84.94, 133.425, 
            82.507, 46.07, 60.097, 44.094, 58.12, 58.12, 137.4, 120.9, 170.9, 
            62.501, 84.94, 133.425, 82.507, 46.07, 60.097]
        
        c = [123.7, 31.7, 45.7, 14.7, 84.7, 27.7, 49.7, 7.1, 2.1, 17.7, 0.85, 0.64]
        d = [31.244, 36.12, 34.784, 92.7, 82.7, 91.6, 56.708, 82.7, 80.8, 
            64.517, 49.4, 49.1]
        e = [0.1, 0.3, 0.4, 0.3, 0.6, 0.3]

        k = 0.7302 * 530 * (14.7 / 40)

        g = np.zeros(20)  # 6 g_i + 15 h_i

        sum_xj = sum(x)

        # g_i para i=1,2,3
        g[0] = (x[0] + x[12]) / (sum_xj + e[0]) - 1
        g[1] = (x[1] + x[13]) / (sum_xj + e[1]) - 1
        g[2] = (x[2] + x[14]) / (sum_xj + e[2]) - 1

        # g_i para i=4,5,6
        g[3] = (x[6] + x[15]) / (sum_xj + e[3]) - 1
        g[4] = (x[7] + x[16]) / (sum_xj + e[4]) - 1
        g[5] = (x[8] + x[17]) / (sum_xj + e[5]) - 1

        # h_i para i=1,...,12
        for i in range(12):
            sum1 = sum(x[j] / b[j] for j in range(12, 24))
            sum2 = sum(x[j] / b[j] for j in range(24))
            h_i = x[i + 12] / (b[i + 12] * sum1) - c[i] * x[i] / (40 * b[i] * sum2)
            g[6 + i] = np.abs(h_i)

        # h13
        h13 = sum(x) - 1
        g[18] = np.abs(h13)

        # h14
        sum1 = sum(x[i] / d[i] for i in range(12))
        sum2 = sum(x[i] / b[i] for i in range(12, 24))
        h14 = sum1 + k * sum2 - 1.671
        g[19] = np.abs(h14)


        return g

    elif problema_id == 20:
        g = np.zeros(6)

        g[0] = -x[0] + 35.0 * np.power(x[1], 0.6) + 35.0 * np.power(x[2], 0.6)
        g[1] = np.abs(-300.0*x[2] + 7500.0*x[4] - 7500.0*x[5] - 25.0*x[3]*x[4]
                      + 25.0*x[3]*x[5] + x[2]*x[3])
        g[2] = np.abs(100.0*x[1] + 155.365*x[3] + 2500.0*x[6] - x[1]*x[3] - 25.0*x[3]*x[6] - 15536.5)
        g[3] = np.abs(-x[4] + np.log(-x[3] + 900.0))
        g[4] = np.abs(-x[5] + np.log(x[3] + 300.0))
        g[5] = np.abs(-x[6] + np.log(-2.0*x[3] + 700.0))

        return g

    elif problema_id == 21:
            g = np.zeros(20) 
            g[0] = -x[0] + 35.0 * np.power(x[1], 0.6) + 35.0 * np.power(x[2], 0.6) + 35.0 * np.power(x[3], 0.6)
            g[1] = np.abs(x[4] - 100000.0*x[7] + 1.0e7)
            g[2] = np.abs(x[5] + 100000.0*x[7] - 100000.0*x[8])
            g[3] = np.abs(x[6] + 100000.0*x[8] - 5.0e7)
            g[4] = np.abs(x[4] + 100000.0*x[9] - 3.3e7) 
            g[5] = np.abs(x[5] + 100000.0*x[10] - 4.4e7)
            g[6] = np.abs(x[6] + 100000.0*x[11] - 6.6e7)
            g[7] = np.abs(x[4] - 120.0*x[1]*x[12]) 
            g[8] = np.abs(x[5] - 80.0*x[2]*x[13]) 
            g[9] = np.abs(x[6] - 40.0*x[3]*x[14]) 
            g[10] = np.abs(x[7] - x[10] + x[15]) 
            g[11] = np.abs(x[8] - x[11] + x[16]) 
            g[12] = np.abs(-x[17] + np.log(x[9] - 100.0)) 
            g[13] = np.abs(-x[18] + np.log(-x[7] + 300.0))
            g[14] = np.abs(-x[19] + np.log(x[15])) 
            g[15] = np.abs(-x[20] + np.log(-x[8] + 400.0)) 
            g[16] = np.abs(-x[21] + np.log(x[16])) 
            g[17] = np.abs(-x[7] - x[9] + x[12]*x[17] - x[12]*x[18] + 400.0) 
            g[18] = np.abs(x[7] - x[8] - x[10] + x[13]*x[19] - x[13]*x[20] + 400.0) 
            g[19] = np.abs(x[8] - x[11] - 4.60517*x[14] + x[14]*x[21] + 100.0) 
            return g

    elif problema_id == 22:
        g = np.zeros(6) # 2 inequalities + 4 equalities

        # Inequalities g1, g2 (indices 0, 1)
        g[0] = x[8]*x[2] + 0.02*x[5] - 0.025*x[4]
        g[1] = x[8]*x[3] + 0.02*x[6] - 0.015*x[7]

        # Equalities h1-h4 (indices 2 to 5) - Stored as |h|
        g[2] = np.abs(x[0] + x[1] - x[2] - x[3])
        g[3] = np.abs(0.03*x[0] + 0.01*x[1] - x[8]*(x[2] + x[3]))
        g[4] = np.abs(x[2] + x[5] - x[4])
        g[5] = np.abs(x[3] + x[6] - x[7])

        return g
    
    elif problema_id == 23:
        g = np.zeros(2)
        g[0] = -2.0*np.power(x[0], 4) + 8.0*np.power(x[0], 3) - 8.0*np.power(x[0], 2) + x[1] - 2.0
        g[1] = -4.0*np.power(x[0], 4) + 32.0*np.power(x[0], 3) - 88.0*np.power(x[0], 2) + 96.0*x[0] + x[1] - 36.0
        return g
    
    else:
        return None
    
def obtener_parametros(problema_id=0):
    if problema_id == 0:

        D = 13
        limites_inf = np.zeros(D)
        limites_sup = np.ones(D)
        limites_sup[9:12] = 100 
        return D, limites_inf, limites_sup
        
    elif problema_id == 1:
        D = 20
        limites_inf = np.zeros(D)
        limites_sup = np.ones(D) * 10
        return D, limites_inf, limites_sup

    elif problema_id == 2:
        D = 10 
        limites_inf = np.zeros(D)    # 0 <= xi
        limites_sup = np.ones(D)     # xi <= 1
        return D, limites_inf, limites_sup


    elif problema_id == 3:
        D = 5 
        limites_inf = np.array([78.0, 33.0, 27.0, 27.0, 27.0])
        limites_sup = np.array([102.0, 45.0, 45.0, 45.0, 45.0])
        return D, limites_inf, limites_sup
    
    elif problema_id == 4:
        D = 4  
        limites_inf = np.array([0.0, 0.0, -0.55, -0.55])
        limites_sup = np.array([1200.0, 1200.0, 0.55, 0.55])
        return D, limites_inf, limites_sup
        
    elif problema_id == 5:
        D = 2 
        limites_inf = np.array([13.0, 0.0])
        limites_sup = np.array([100.0, 100.0])
        return D, limites_inf, limites_sup
            
    elif problema_id == 6:
        D = 10 
        limites_inf = np.ones(D) * -10.0
        limites_sup = np.ones(D) * 10.0
        return D, limites_inf, limites_sup

    elif problema_id == 7:
        D = 2
        limites_inf = np.array([0.0, 0.0])
        limites_sup = np.array([10.0, 10.0])
        return D, limites_inf, limites_sup

    elif problema_id == 8:
        D = 7
        limites_inf = np.ones(D) * -10.0
        limites_sup = np.ones(D) * 10.0
        return D, limites_inf, limites_sup
        
    elif problema_id == 9:
        D = 8
        limites_inf = np.array([100.0, 1000.0, 1000.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        limites_sup = np.array([10000.0, 10000.0, 10000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
        return D, limites_inf, limites_sup

    elif problema_id == 10:
        D = 2
        limites_inf = np.array([-1.0, -1.0])
        limites_sup = np.array([1.0, 1.0])
        return D, limites_inf, limites_sup

    elif problema_id == 11:
        D = 3
        limites_inf = np.array([0.0, 0.0, 0.0])
        limites_sup = np.array([10.0, 10.0, 10.0])
        return D, limites_inf, limites_sup

    elif problema_id == 12:
        D = 5
        limites_inf = np.array([-2.3, -2.3, -3.2, -3.2, -3.2])
        limites_sup = np.array([2.3, 2.3, 3.2, 3.2, 3.2])
        return D, limites_inf, limites_sup

    elif problema_id == 13:
        D = 10
        limites_inf = np.zeros(D)
        limites_sup = np.ones(D) * 10.0
        return D, limites_inf, limites_sup

    elif problema_id == 14:
        D = 3
        limites_inf = np.zeros(D)
        limites_sup = np.ones(D) * 10.0
        return D, limites_inf, limites_sup

    elif problema_id == 15: 
        D = 5  
        limites_inf = np.array([704.4148, 68.6, 0, 193, 25])
        limites_sup = np.array([906.3855, 288.88, 134.75, 287.0966, 84.1988])
        return D, limites_inf, limites_sup

    elif problema_id == 16:
        D = 6  # Dimensión 6 para variables x1 a x6
        limites_inf = np.array([0.0, 0.0, 340.0, 340.0, -1000.0, 0.0])
        limites_sup = np.array([400.0, 1000.0, 420.0, 420.0, 1000.0, 0.5236])
        return D, limites_inf, limites_sup

    elif problema_id == 17:
        D = 9

        limites_inf = np.ones(D) * -10.0 
        limites_sup = np.ones(D) * 10.0  
        
        limites_inf[8] = 0.0  
        limites_sup[8] = 20.0 
        return D, limites_inf, limites_sup

    elif problema_id == 18:
        D = 15
        limites_inf = np.zeros(D)
        limites_sup = np.ones(D) * 10.0
        
        return D, limites_inf, limites_sup

    elif problema_id == 19:
        D = 24
        limites_inf = np.zeros(D)
        limites_sup = np.ones(D) * 10.0
    
        return D, limites_inf, limites_sup

    elif problema_id == 20:
        D = 7
        limites_inf = np.array([0.0, 0.0, 0.0, 100.0, 6.3, 5.9, 4.5])
        limites_sup = np.array([1000.0, 40.0, 40.0, 300.0, 6.7, 6.4, 6.25])
        return D, limites_inf, limites_sup

    elif problema_id == 21:
            D = 22 
            limites_inf = np.array([0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 100.0, 100.0, 100.01, 
                                    100.0, 100.0, 0.0, 0.0, 0.0, 0.01, 0.01, 4.7, 4.7, 
                                    4.7, 4.7, 4.7, 4.7])
            limites_sup = np.array([20000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 4.0e7, 299.99, 300.0, 
                                    400.0, 600.0, 500.0, 600.0, 1000.0, 300.0, 400.0, 6.25, 6.25, 
                                    6.25, 6.25, 6.25, 6.25])
            return D, limites_inf, limites_sup

    elif problema_id == 22:
        D = 9
        limites_inf = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01])
        limites_sup = np.array([300.0, 300.0, 100.0, 200.0, 100.0, 300.0, 100.0, 200.0, 0.03])
        return D, limites_inf, limites_sup

    elif problema_id == 23:
        D = 2
        limites_inf = np.array([0.0, 0.0])
        limites_sup = np.array([3.0, 4.0])
        return D, limites_inf, limites_sup
    
    else:
        return None, None, None
