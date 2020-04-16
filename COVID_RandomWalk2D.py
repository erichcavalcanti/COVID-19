import numpy as np
#import matplotlib.animation as mani
import matplotlib.pyplot as plt

# ---- FUNCOES DE EVOLUCAO

def dar_um_passo(PESSOAS,L,PASSO=1,FLAG_CONTORNO=1):
    """ 
    Incrementa com um passo aleatório todo um conjunto de pessoas.\n
    Utilizada condição de contorno periodica nas bordas.
    
    INPUT:
        PESSOAS, conjuntos de pessoas;\n
        L, tamanho da rede, para garantir que passos estão dentro da rede;\n
        PASSO, tamanho do passo, por padrão 1  
        FLAG_CONTORNO: (por padrão, 1)
            0 Periodica
            1 Refletora nas bordas
    """
    
#    # Caso com posições na reta inteira
#    PESSOAS  += np.random.randint(-PASSO,PASSO+1,size=PESSOAS.shape) 
#    PESSOAS %= L #condição de contorno periodica

    # Posições na reta real
    
    # direção da movimentação
    Theta = np.random.random(size= PESSOAS.T[0].size)
    # adiciona passo com tamanho estabelecido
    PESSOAS  += np.append([PASSO*np.sin(Theta)],[PASSO*np.cos(Theta)],axis=0).T

    #condição de contorno periodica para região restrita ao primeiro quadrante do plano cartesiano
    if FLAG_CONTORNO==0 :
        PESSOAS %= L 
    
    #condição de contorno de borda refletora
    if FLAG_CONTORNO==1 :
        #quem estiver fora do quadrante um é refletido para o quadrante um
        PESSOAS = np.where(PESSOAS<0,-PESSOAS,PESSOAS)
        #quem estiver dentro do quadrante um, mas fora da região permitida, é colocado para dentro
        PESSOAS = np.where(PESSOAS>L,2*L-PESSOAS,PESSOAS)
        
    return PESSOAS

def check_mudanca(CONJUNTO1,CONJUNTO2,PROB=.5):#otimizar
    """
    Pessoa do CONJUNTO1 pode ser transformada em Pessoa do CONJUNTO2 com probabilidade PROB
    
    INPUT:
        CONJUNTO1, ;\n
        CONJUNTO2, ;\n
        PROB, por padrao 100%
    
    OUTPUT:
        CONJUNTO1, ;\n
        CONJUNTO2, .
    """
    
    'MASK INDICARÁ ELEMENTOS A SEREM MODIFICADOS COM CADEIRA TRUE/FALSES'
    MASK_DEL = np.array(np.random.random(CONJUNTO1.T[0].size) < PROB,dtype=bool)
    
    if (MASK_DEL.sum()>0):
        CONJUNTO2 = np.append(CONJUNTO2, CONJUNTO1[MASK_DEL],axis=0)
        CONJUNTO1 = CONJUNTO1[np.bitwise_not(MASK_DEL)] 
#        AUX = np.arange(MASK_DEL.size)
#        CONJUNTO1 = np.delete(CONJUNTO1, AUX[MASK_DEL], axis=0)
        
    return CONJUNTO1,CONJUNTO2

def check_colisao(EL1,EL2,PROB=1,PROX=1.):
    """ 
    Confere se pessoa do CONJUNTO1 colide com alguem do CONJUNTO2.
    Se colisao ocorre pode ocorrer transformaçao de um para outro.
    
    Entrada:
        EL1, primeiro conjunto de pessoas;\n
        EL2, segundo conjunto de pessoas;\n
        PROX, proximidade necessária. Padrão 1.1;\n
        PROB, probabilidade de transformar o estado se estivar na proximidade suposta. Padrão 1
    
    
    ####
    Cuidado do código
        - loop nos EL1 avaliando se algum EL2 é próximo (medida de distância)
        - ao detetar proximidade, elemento de EL2 precisa se convertido para EL1.
        Remover de EL2. Adicionar em EL1.
        A remoção de EL2 deve ser feita após loop de EL2.
        A adição em EL1 deve ser feita após loop de EL1.
        Para tal, usaremos varaveis auxiliares.    
    """    
    NEW_EL2 = np.copy(EL2)

    'para cada EL2'
    for i in range(EL2.T[0].size):
        'comparar com EL1'        
        
        'para registrar elementos de EL1 que serão removidos após loop'
        MASK_DEL1=np.array(np.random.random(EL1.T[0].size) < PROB,dtype=bool)
        'mede distância um elemento de EL2 e todos de EL1'
        AUX_DIFF = (EL2[i]-EL1)*(EL2[i]-EL1)
        'termina o produto interno'
        DIFF = np.sqrt( AUX_DIFF.T[0] + AUX_DIFF.T[1] )
        'gera mascara'
        MASK_DIFF = np.array(DIFF<PROX, dtype=bool)        
        
        'serão manipulados SOMENTE elementos que satisfizerem ambas mascaras'
        MASK = np.bitwise_and(MASK_DEL1,MASK_DIFF)
        if (MASK.sum()>0):
            'adiciona EL2 em uma cadeia auxiliar (para não atrapalhar for)'
            NEW_EL2 = np.append(NEW_EL2, EL1[MASK],axis=0)
            'atualiza EL1 removendo alguns elementos'
            EL1 = EL1[np.bitwise_not(MASK)] 
#            AUX_DEL = np.arange(MASK.size)
#            EL1 = np.delete(EL1,AUX_DEL[MASK],axis=0)

    'após sair do loop dos EL2s, atualizar quem são os EL2s'
    EL2 = np.copy(NEW_EL2)

    return EL1,EL2

def modelo_SEIR2(L,NB,STEPS,P_E,P_I,P_R,Conf_Ini,Conf_Fim):
    """
    L - rede quadrada L*L. Configurado para L impar.
    NB - numero de caminhantes bebados no sistema
    STEPS - numero de passos
    P_E - probabilidade de infecção ao ter contato, classificado como exposo sem sintomas
    P_I - probabilidade de apresentar sintomas e ser isolado.
    P_R - probabilidade de cura espontanea (com imunidade)
    
    trabalha-se em um plano xy, quadrante 1 somente.
    imprime-se convertendo para notação de indices linha-coluna.

    """
    'inicializando rede'
    
    'distribuindo as posições de cada elemento'
    e_saudavel = L*np.random.random(size=(NB-1,2)) #NB-1 saudavel
    e_exposto = L*np.random.random(size=(1,2)) #1 assintomatico
    e_infectado = L*np.random.random(size=(0,2)) #0 sintomaticos
    e_ISOLADO = L*np.random.random(size=(0,2)) #0 isolados
    e_recuperados = L*np.random.random(size=(0,2)) #0 recuperados
    
    ' acompanhando infecções '
    dados = np.array([ [e_saudavel.T[0].size],    # casos saudavel
                        [e_exposto.T[0].size+e_infectado.T[0].size],
                        [e_ISOLADO.T[0].size],
                        [e_recuperados.T[0].size]])  
    
    
    for i in range(STEPS):      
        'dar um passo'
        e_saudavel = dar_um_passo(e_saudavel,L)
        e_exposto = dar_um_passo(e_exposto,L)
        e_infectado = dar_um_passo(e_infectado,L)
        #e_recuperados = dar_um_passo(e_recuperados,L) 
        ##movimento de recuperados não afeta o sistema, por não ser ciclico
        'contagio por contato? mesma taxa para assintomaticos ou não'
        e_saudavel, e_exposto = check_colisao(e_saudavel, e_exposto,P_E)
        e_saudavel, e_infectado = check_colisao(e_saudavel, e_infectado,P_E)
        'começa a apresentar sintomas?'
        e_exposto,e_infectado = check_mudanca(e_exposto,e_infectado,P_I)
        'no periodo de isolamento para pessoas sintomaticas'
        if (i>Conf_Ini and i<Conf_Fim):
            'testa se sintomatico será confinado'
            e_infectado,e_ISOLADO = check_mudanca(e_infectado,e_ISOLADO,.9)
        'ao fim  do isolamento'
        if (i>Conf_Fim):
            'libera isolados'
            e_ISOLADO,e_infectado = check_mudanca(e_ISOLADO,e_infectado,1)
            
        'infectado se recupera?'
        e_ISOLADO,e_recuperados = check_mudanca(e_ISOLADO,e_recuperados,P_R)
        e_exposto,e_recuperados = check_mudanca(e_exposto,e_recuperados,P_R)
        e_infectado,e_recuperados = check_mudanca(e_infectado,e_recuperados,P_R)
        'guardar dados'
        aux = np.array([ [e_saudavel.T[0].size],[e_exposto.T[0].size+e_infectado.T[0].size],[e_ISOLADO.T[0].size],[e_recuperados.T[0].size]])
        dados = np.append(dados,aux,axis=1)
    
    return dados

def estat_SEIR2(L,NB,STEPS,P_E,P_I,P_R,SAMPLES,Conf_Ini,Conf_Fim):
    """
    L - rede quadrada L*L. Configurado para L impar.
    NB - numero de caminhantes bebados no sistema
    STEPS - numero de passos
    P_E - probabilidade de infecção ao ter contato, classificado como exposo sem sintomas
    P_I - probabilidade de apresentar sintomas e ser isolado.
    P_R - probabilidade de cura espontanea (com imunidade)
    SAMPLES - numero de amostrar para fazer a estatistica
    """
    DATA = modelo_SEIR2(L,NB,STEPS,P_E,P_I,P_R,Conf_Ini,Conf_Fim)

    for i in range(1,SAMPLES):
        DATA += modelo_SEIR2(L,NB,STEPS,P_E,P_I,P_R,Conf_Ini,Conf_Fim)
    
    DATA = DATA/(SAMPLES*NB)
    
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax.plot(DATA[0],'b--',label='Saudavel')
    ax.plot(DATA[1],'r--',label='Exposto/Infectado')
    ax.plot(DATA[2],'g--',label='Isolado')
    ax.plot(DATA[3],'y--',label='Recuperado')
    ax.legend(loc='upper right')
    ax.set_xlabel('Dias')
    ax.set_ylabel('População (%)')
    fig.savefig("RW_SEIR_Epidemia.png")
    
    return DATA

'sem isolamento, temos um cenario desastroso em que todos estão expostos'
a = estat_SEIR2(L=10,NB=75,STEPS=100,P_E=1,P_I=.3,P_R=.01,SAMPLES=100,Conf_Ini=0,Conf_Fim=0)
'isolando alguns somente os casos que apresenta sintomas, e supondo erro nesse'
'processo, as coisas não melhoram'
a = estat_SEIR2(10,75,100,1,.3,.01,100,1,20)

"""
Isolamento quase perfeito daqueles que apresentam sintomas
    a = estat_SEIR2(10,75,100,1,.9,.01,100,1,20)

"""
a = estat_SEIR2(10,75,100,1,.9,.01,100,1,100)

a = estat_SEIR2(10,25,200,1,1,.01,100,100,100)
a = estat_SEIR2(2,1,10,1,1,.1,200,100,100)
