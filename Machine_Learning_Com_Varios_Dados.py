#!/usr/bin/env python
# coding: utf-8

# # Notebook realizado por Marina Lima no curso de Machine Learning.

# 1. Segregação dos Dados

# In[1]:


import pandas as pd


# In[2]:


resultados_exames = pd.read_csv('dados/exames.csv')


# In[3]:


resultados_exames


# In[4]:


resultados_exames.head()


# In[5]:


from sklearn.model_selection import train_test_split
from numpy import random

SEED = 123143
random.seed(SEED)

valores_exames = resultados_exames.drop(columns=['id', 'diagnostico'])
diagnostico = resultados_exames.diagnostico

treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames, diagnostico)


# In[6]:


treino_x.head()


# In[7]:


treino_y.head()


# 2. Criar modelo de Classificação

# Para obtermos o resultado aplicado aos dados de teste, utilizaremos o método score(). Esse método recebe os dados de teste (teste_x e teste_y) e nos retorna uma acurácia.

# In[12]:


from sklearn.ensemble import RandomForestClassifier

classificador = RandomForestClassifier(n_estimators = 100) #n estimators diz quantas arvores de decisão ele vai construir
classificador.fit(treino_x, treino_y)

print(classificador.score(teste_x, teste_y))


# Deu esse erro porque o RandomForest não aceita valores vazios, e há valores vazios na database, então precisamos retirar esses valores.

# O Pandas possui uma função chamada isnull(), que retorna true para todas as células não preenchidas de um conjunto de dados, e false para as preenchidas.

# In[13]:


resultados_exames.isnull()


# Juntando isnull().sum() ele vai retornar as colunas que tem valores vazios.

# In[14]:


resultados_exames.isnull().sum()


# In[15]:


419/569


# In[23]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from numpy import random

SEED = 123143
random.seed(SEED)

valores_exames = resultados_exames.drop(columns=["id", "diagnostico"])
diagnostico = resultados_exames.diagnostico
valores_exames_v1 = valores_exames.drop(columns="exame_33")

treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v1, 
                                                        diagnostico, test_size = 0.3)


classificador = RandomForestClassifier(n_estimators = 100)
classificador.fit(treino_x, treino_y)
print("Resultado da classificação %.2f%%" % (classificador.score(teste_x, teste_y)* 100))


# In[25]:


from sklearn.dummy import DummyClassifier

SEED = 123143
random.seed(SEED)

classificador_bobo = DummyClassifier(strategy = "most_frequent")
classificador_bobo.fit(treino_x, treino_y)
print("Resultado da classificação boba %.2f%%" % (classificador_bobo.score(teste_x, teste_y)* 100))


# Definimos a baseline como 92.40%

# Agora que determinamos nosso baseline como 92.40%, precisamos estudar o comportamento das variáveis, entendendo quais são os valores de "exame_1", por exemplo, que caracterizam um resultado como benigno ou maligno. Estudar o comportamento dos dados é muito mais fácil quando trabalhamos com visualizações, e faremos isso com o ViolinPlot, um gráfico em formato de violino.

# Usaremos a função melt() do Panda, que consegue pegar um dataframe e transformá-lo em uma tabela contendo as variáveis, os valores dessas variáveis e as classes pertencentes a elas.

# In[29]:


dados_plot = pd.melt(dados_plot, id_vars="diagnostico", var_name="exames", value_name="valores")
dados_plot.head()


# In[37]:


import seaborn as sns
import matplotlib.pyplot as plt

dados_plot = pd.concat([diagnostico, valores_exames_v1.iloc[:,0:10]], axis = 1)
dados_plot = pd.melt(dados_plot, id_vars="diagnostico", var_name="exames", value_name="valores")

plt.figure(figsize=(10,10))
sns.violinplot(x = "exames", y = "valores", 
               hue = "diagnostico", data = dados_plot) #HUE - classe maligno ou benigno

plt.xticks(rotation = 90)


# Precisamos encontrar um modo de padronizarmos o eixo Y do nosso gráfico de modo a torná-lo analisável. Para isso, usaremos o StandarScaler do SKlearn, que padroniza os dados de acordo com uma função matemática.

# In[44]:


from sklearn.preprocessing import StandardScaler

padronizador = StandardScaler()
padronizador.fit(valores_exames_v1)
valores_exames_v2 = padronizador.transform(valores_exames_v1)
valores_exames_v2


# Essa padronização limitará o eixo Y, tornando possível a análise. Então, importaremos o StandardScaler do módulo sklearn.preprocessing e o instanciaremos em uma variável padronizador. Em seguida, adaptaremos o padronizador aos nossos dados, valores_exames_v1, com fit(), e faremos uma transformação desses dados com transform(), guardando o resultado em outra variável, valores_exames_v2.
# 
# Agora, na concatenação, passaremos os valores_exames_v2 ao invés de valores_exames_v1. Assim, melhoraremos nossa visualização. Porém, o retorno de transform é um array do Numpy, e no concat() estamos trabalhando com um dataframe. Para resolvermos isso, executaremos pd.DataFrame(), passando os dados com data = valores_exames_v2 e as colunas com columns = valores_exames_v1.keys().

# In[50]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

padronizador = StandardScaler()
padronizador.fit(valores_exames_v1)
valores_exames_v2 = padronizador.transform(valores_exames_v1)
valores_exames_v2 = pd.DataFrame(data = valores_exames_v2, columns=valores_exames_v1.keys())

dados_plot = pd.concat([diagnostico, valores_exames_v2.iloc[:,0:10]], axis = 1)
dados_plot = pd.melt(dados_plot, id_vars="diagnostico", var_name="exames", value_name="valores")

plt.figure(figsize=(10,10))
sns.violinplot(x = "exames", y = "valores", 
               hue = "diagnostico", data = dados_plot, split=True) #HUE - classe maligno ou benigno

plt.xticks(rotation = 90)


# Agora que geramos nosso gráfico, começaremos a analisá-lo. No eixo "x" temos os exames, no "y" os valores normalizados, e no corpo do gráfico temos o violin plot. Mas como interpretar essas informações? Como as legendas apontam, do lado esquerdo, em azul, temos os cânceres do tipo maligno, e do lado direito, em laranja, os do tipo benigno. A parte mais alta da curva são os valores que mais ocorrem para cada tipo.

# In[52]:


valores_exames_v1.exame_4


#  O "exame_4", que é o que mais chama atenção, já que é a única reta no nosso gráfico! Mas por que isso acontece? Se exibirmos os valores dessa coluna com valores_exames_v2.exame_4, perceberemos que todos os resultados são 1.0. Se ao invés disso utilizarmos valores_exames_v1.exame_4, veremos que essa coluna recebe um valor constante 103.78. Mas o que valores constantes agregam ao nosso conjunto? Não.
#  
#  Voltando ao nosso conjunto de dados, agora sabemos que podemos eliminar features como "exame_4", já que temos um valor constante.

# In[56]:


def grafico_violino(valores, inicio, fim):

    dados_plot = pd.concat([diagnostico, valores.iloc[:,inicio:fim]], axis = 1)
    dados_plot = pd.melt(dados_plot, id_vars="diagnostico", 
                         var_name="exames",
                         value_name="valores")

    plt.figure(figsize=(10,10))

    sns.violinplot(x = "exames", y = "valores", hue = "diagnostico", 
                    data = dados_plot, split = True)

    plt.xticks(rotation = 90)

grafico_violino(valores_exames_v2, 10, 21)


# In[57]:


grafico_violino(valores_exames_v2, 21, 32)


# In[88]:


valores_exames_v3 = valores_exames_v2.drop(columns=["exame_29","exame_4"])

def classificar(valores):
    SEED = 1234
    random.seed(SEED)

    treino_x, teste_x, treino_y, teste_y = train_test_split(valores, diagnostico, test_size = 0.3)

    classificador = RandomForestClassifier(n_estimators = 100)
    classificador.fit(treino_x, treino_y)
    print("Resultado da classificação %.2f%%" % (classificador.score(teste_x, teste_y)* 100))
    
classificar(valores_exames_v3)


# Como resultado, teremos 92.98% - ou seja, com a remoção das constantes (a redução de 2 dimensões), tivemos um pequeno aumento na acurácia do nosso classificador em relação à baseline que definimos anteriormente.

# # Calcular a correlação

# O próximo objetivo é calcularmos a correlação a partir do nosso dataframe valores_exames_v3, o que é possível com a função corr() do Pandas, que nos retornará justamente a matriz de correlação (a correlação entre todas as variáveis).

# In[60]:


valores_exames_v3


# In[61]:


valores_exames_v3.corr() 


#  Na função heatmap(), passaremos também alguns parâmetros que poderão nos ajudar na interpretação dessa matriz, começando pelo annot = True, que anotará em cada quadrado o valor da correlação. O outro é fmt = ".1f", que nos mostrará apenas uma casa decimal do valor exibido na célula.

# In[73]:


plt.figure(figsize = (17,15))
sns.heatmap(matriz_correlacao, annot = True, fmt = ".1f") #apenas uma casa decimal


# Na diagonal principal, temos valores em branco (1,0) que são totalmente correlacionados, afinal fazem a correspondência de uma feature com ela mesma. Sendo assim, não é muito do nosso interesse trabalhar com ela, mas sim com as outras features da matriz.
# 
# Por exemplo, é possível encontrar diversas features totalmente correlacionadas, com valores iguais a 1.0, e outras com um valor bem próximo, como 0.9. Enquanto isso, também temos features pouquíssimo correlacionadas, com valores próximos a 0.0. Por enquanto trabalharemos na remoção das features altamente correlacionadas.
# 
# Criaremos então uma matriz_correlacao_v1 que armazenará o retorno de todos as correlações superiores a 0.99. ou seja, que possuem uma correlação quase perfeita.

# In[74]:


matriz_correlacao_v1 = matriz_correlacao[matriz_correlacao>0.99]
matriz_correlacao_v1


# In[77]:


matriz_correlacao_v2 = matriz_correlacao_v1.sum()


# In[78]:


matriz_correlacao_v2


# In[79]:


variaveis_correlacionadas = matriz_correlacao_v2[matriz_correlacao_v2>1]
variaveis_correlacionadas


# In[89]:


valores_exames_v4 = valores_exames_v3.drop(columns=variaveis_correlacionadas.keys())


# In[91]:


valores_exames_v4.head()


# In[92]:


classificar(valores_exames_v4)


# In[93]:


valores_exames_v5 = valores_exames_v3.drop(columns=["exame_3", "exame_24"])
classificar(valores_exames_v5)


# Agora que excluímos as features de alta correlação do nosso dataset, será que existe uma maneira de selecionarmos um determinado número ("k") de melhores features desse conjunto? A ideia seria gerarmos uma pontuação para cada feature (cada exame) e selecionar as melhores dentre elas.

# O SKlearn possui um método SelectKBest() que faz justamente isso: a partir de uma função matemática, ele gera um score para cada feature e seleciona um determinado número de features dentre os melhores scores. Para nossa sorte, implementar o SelectKBest é relativamente simples. Começaremos importando o SelectKBest do módulo sklearn.feature_selection.
# 
# Na chamada, além de um número k de features, precisaremos passar a função matemática que deverá ser utilizada. **Uma função muito utilizada é o Qui-quadrado, que usaremos para inferir quais features serão mais representativas para nosso processo de classificação. A partir do módulo sklearn.feature_selection, importaremos também a função chi2, que passaremos como parâmetro de SelectKBest().**
# 
# **No nosso caso, utilizaremos 5 como valor de k.**
# 
# Ou seja, de exames que eram realizados para chegar a um diagnóstico, queremos realizar apenas 5, mantendo a qualidade de predição do nosso algoritmo. Armazenaremos o retorno dessa chamada em uma variável selecionar_kmelhores e exibiremos o seu conteúdo na tela.

# In[94]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

selecionar_kmelhores = SelectKBest(chi2, k = 5)


# In[95]:


selecionar_kmelhores


# Agora precisamos treinar nosso modelo e realizar a transformação dos dados. Para isso, precisaremos separar novamente os dados de treino e de teste, o que dessa vez será feito com o conjunto valores_exames_v5. Então, usaremos o fit() para treinarmos o modelo selecionar_kmelhores com os dados treino_x e treino_y.

# In[96]:


treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v5, 
                                                        diagnostico,
                                                        test_size = 0.3)



selecionar_kmelhores.fit(treino_x,treino_y)


# **OBS:** 
# A chi quadrada não aceita valores negativos, e existem vários deles, por exemplo, na coluna "exame_5". Uma alternativa para corrigirmos isso seria voltarmos para os valores não-normalizados que tínhamos no conjunto valores_exames_v1. Sendo assim, teremos que remover deste conjunto todas as colunas que não estão presentes em valores_exames_v5. Para isso, faremos um drop() das colunas "exame_4", "exame_29", "exame_3" e "exame_24", e armazenaremos o retorno em uma variável valores_exames_v6.

# In[97]:


valores_exames_v6 = valores_exames_v1.drop(columns=(["exame_4", "exame_29", "exame_3", "exame_24"]))


# In[100]:


SEED= 1234                    #seed = variável de aleatoriedade
random.seed(SEED)

treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v6, 
                                                        diagnostico,
                                                        test_size = 0.3)



selecionar_kmelhores.fit(treino_x,treino_y)
treino_kbest = selecionar_kmelhores.transform(treino_x)
teste_kbest = selecionar_kmelhores.transform(teste_x)


# In[101]:


teste_kbest.shape


# In[102]:


classificador = RandomForestClassifier(n_estimators=100, random_state=1234)
classificador.fit(treino_kbest, treino_y)
print("Resultado da classificação %.2f%%" %(classificador.score(teste_kbest,teste_y)*100))


# In[111]:


5/33


# Mas será que analisar somente a acurácia é o suficiente, ou seria mais interessante trabalharmos mais detalhadamente na análise dos nossos dados?
# 
# Pensando nisso, a seguir aprenderemos outra forma de analisarmos os resultados de classificação.

# ## Matriz de confusão

# **Matriz de confusão, representada no SKlearn pela função confusion_matrix(). Essa função nos retorna uma matriz na qual os elementos i são os valores reais e os elementos j são os valores de predição. Como parâmetros, ela recebe os valores reais (y_true) e os valores preditos (y_pred).**
# 
# No nosso projeto, importaremos a função confusion_matrix do módulo sklearn.metrics e criaremos uma variável matriz_confusao que guardará o retorno da sua chamada. O primeiro parâmetro que passaremos para essa função é o teste_y, que compõe os resultados reais. Também **precisaremos passar as predições, mas não as temos ainda, afinal estávamos calculando apenas a acurácia.**
# 
# **Para obtermos as predições, chamaremos classificador.predict() com teste_kbest como parâmetro - afinal, treinamos esse classificador com os dados treino_kbest selecionados pelo modelo select_kbest().**

# In[103]:


from sklearn.metrics import confusion_matrix

matriz_confusao = confusion_matrix(teste_y,classificador.predict(teste_kbest))


# In[104]:


matriz_confusao


# In[105]:


plt.figure(figsize = (10, 8))
sns.set(font_scale= 2)
sns.heatmap(matriz_confusao, annot = True, fmt = "d").set(xlabel = "Predição", ylabel= "Real")


# Mas como interpretá-la? O eixo Y está representando nossos valores reais, e o eixo X representa os valores de predição. Temos 0 quando o diagnóstico é de um câncer benigno e 1 quando é de um câncer maligno. A soma dos dois quadrados superiores nos trará o total de cânceres diagnosticados como benignos, e a dos dois quadrados inferiores o total de cânceres diagnosticados como malignos.
# 
# De 105 casos benignos, nosso modelo acertou 100, classificando o restante (5) como maligno. Já nos casos de cânceres malignos, nosso modelo acertou 58, classificando 8 como benignos. Mas por que essas informações são importantes?
# 
# Em alguns casos, como na área da saúde, é muito importante sabermos qual classificação estamos acertando mais. Imagine, por exemplo, uma pessoa que realmente tem câncer, mas recebe o diagnóstico de que não tem. Esse tipo de equívoco no diagnóstico tornaria o tratamento mais difícil, afinal o tempo de realização impacta nas chances de cura desse tipo de doença.

# ##  Seleção com RFE
# 
# Nessa nova técnica, por meio da acurácia, que será nossa forma de avaliação, nosso classificador saberá qual das features é mais importante pra ele, descartando as de menor acurácia. Esse processo será feito sucessivamente até atingir o número de features selecionado - no no nosso caso, 5. Esse modelo, que implementaremos em Python, é chamado de RFE - Recursive Feature Elimination, algo como "Eliminação de Feature por Recursão".

# In[106]:


from sklearn.feature_selection import RFE

SEED= 1234
random.seed(SEED)

treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v6, 
                                                        diagnostico,
                                                        test_size = 0.3)

classificador = RandomForestClassifier(n_estimators=100, random_state=1234)
classificador.fit(treino_x, treino_y)
selecionador_rfe = RFE(estimator = classificador, n_features_to_select = 5, step = 1)
selecionador_rfe.fit(treino_x, treino_y)
treino_rfe = selecionador_rfe.transform(treino_x)
teste_rfe = selecionador_rfe.transform(teste_x)
classificador.fit(treino_rfe, treino_y)

matriz_confusao = confusion_matrix(teste_y,classificador.predict(teste_rfe))
plt.figure(figsize = (10, 8))
sns.set(font_scale= 2)
sns.heatmap(matriz_confusao, annot = True, fmt = "d").set(xlabel = "Predição", ylabel= "Real")

print("Resultado da classificação %.2f%%" %(classificador.score(teste_rfe,teste_y)*100))


# Até o momento nós selecionamos algumas features com base em visualizações, como o Violin Plot e a Matriz de Correlação, e com alguns algoritmos mais automatizados, como o SelectKBest e o RFE. No caso desses algoritmos, nós determinamos quantas features gostaríamos que fossem selecionadas - no nosso caso 5, mas poderiam ser 10, 15 ou qualquer outro número, dependendo da necessidade.
# 
# **A questão agora é: será que existe alguma técnica que nos informa qual conjunto de features gerará o melhor resultado? Essa técnica é o RFE Cross Validation. 
# O RFECV divide o nosso banco de dados em blocos e aplica o algoritmo RFE, que acabamos de aprender, em cada um desses blocos, gerando diferentes resultados. Dessa forma, O RFECV não só nos informa quantas features precisamos ter para gerar o melhor resultado possível, como também quais features são essas.**
# 
# A implementação do RFECV é bastante semelhante à do RFE, portanto vamos copiar o código que criamos anteriormente e fazer as devidas modificações, a começar pela importação do RFECV. Criaremos então um selecionador_rfecv que receberá a chamada de RFECV com o mesmo estimador que criamos antes (nosso classificador), o número de divisões que deverão ser feitas na base de dados (cv, que definiremos como 5) e uma função de avaliação que deverá ser utilizada (scoring, que definiremos como accuracy). Não precisaremos passar passar um número de features, mas mantaremos o número de passos (step = 1).

# In[109]:


from sklearn.feature_selection import RFECV

SEED= 1234
random.seed(SEED)

treino_x, teste_x, treino_y, teste_y = train_test_split(valores_exames_v6, 
                                                        diagnostico,
                                                        test_size = 0.3)

classificador = RandomForestClassifier(n_estimators=100, random_state=1234)
classificador.fit(treino_x, treino_y)
selecionador_rfecv = RFECV(estimator = classificador, cv = 5, step = 1, scoring="accuracy")
selecionador_rfecv.fit(treino_x, treino_y)
treino_rfecv = selecionador_rfecv.transform(treino_x)
teste_rfecv = selecionador_rfecv.transform(teste_x)
classificador.fit(treino_rfecv, treino_y)

matriz_confusao = confusion_matrix(teste_y,classificador.predict(teste_rfecv))
plt.figure(figsize = (10, 8))
sns.set(font_scale= 2)
sns.heatmap(matriz_confusao, annot = True, fmt = "d").set(xlabel = "Predição", ylabel= "Real")

print("Resultado da classificação %.2f%%" %(classificador.score(teste_rfecv,teste_y)*100))


# In[112]:


treino_x.columns[selecionador_rfecv.support_]


# In[113]:


import matplotlib.pyplot as plt

plt.figure(figsize = (14, 8))
plt.xlabel("Número de exames")
plt.ylabel("Acurácia")
plt.plot(range(1, len(selecionador_rfecv.grid_scores_) +1), selecionador_rfecv.grid_scores_)
plt.show()


# In[114]:


valores_exames_v7 = selecionador_rfe.transform(valores_exames_v6)


# In[115]:


valores_exames_v7.shape


# In[116]:


import seaborn as sns
plt.figure(figsize=(14,8))
sns.scatterplot(x = valores_exames_v7[:,0] , y = valores_exames_v7[:,1], hue = diagnostico)


# In[117]:


from sklearn.decomposition import PCA

pca = PCA(n_components = 2)
valores_exames_v8 = pca.fit_transform(valores_exames_v5)
plt.figure(figsize=(14,8))
sns.scatterplot(x = valores_exames_v8[:,0] , y = valores_exames_v8[:,1], hue = diagnostico)


# In[118]:


from sklearn.manifold import TSNE

tsne = TSNE(n_components = 2)
valores_exames_v9 = tsne.fit_transform(valores_exames_v5)
plt.figure(figsize=(14,8))
sns.scatterplot(x = valores_exames_v9[:,0] , y = valores_exames_v9[:,1], hue = diagnostico)


# In[ ]:




