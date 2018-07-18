from sklearn.naive_bayes import MultinomialNB

#parametros analisados
late = 1
nao_late = 0

mia = 1
nao_mia = 0

porte_pequeno = 1
porte_grande = 0

gordinho = 1
magrinho = 0

bigode_curto = 1
bigode_comprido = 0

# lista de características de gatinhos
animal_1 = [porte_pequeno, bigode_comprido, gordinho, mia, nao_late]
animal_2 = [porte_pequeno, bigode_curto,    gordinho, mia, nao_late]  
animal_3 = [porte_pequeno, bigode_comprido, magrinho, mia, nao_late]

# lista de cachorrinhos
animal_4 = [porte_pequeno, bigode_curto, gordinho, nao_mia, late    ]
animal_5 = [porte_grande,  bigode_curto, magrinho, nao_mia, late    ]
animal_6 = [porte_grande,  bigode_curto, gordinho, nao_mia, nao_late]

# tags de classificacoes
gato = 1 
cachorro = -1 

# lista de animais
conjunto_animais = [animal_1, animal_5]
#conjunto_animais = [animal_1, animal_2, animal_3, animal_4, animal_5, animal_6]

# identificacao respectivamente
identificacao = [gato, cachorro]
#identificacao = [gato, gato, gato, cachorro, cachorro, cachorro]

# criando um modelo
modelo = MultinomialNB()

# treinando ele com base nas informações
# características vs identificacao
modelo.fit(conjunto_animais, identificacao)

#animais que eu nao classifiquei ainda
animal_sem_id_1 = [porte_pequeno, bigode_curto, gordinho, mia,      nao_late   ]
animal_sem_id_2 = [porte_pequeno, bigode_curto, magrinho, mia,      late       ]
animal_sem_id_3 = [porte_grande,  bigode_curto, gordinho, nao_mia,  late       ]
animal_sem_id_4 = [porte_pequeno, bigode_curto, gordinho, nao_mia,  nao_late   ]

# colocando os animais nao identificados na mesma lista
conjunto_sem_id = [animal_sem_id_1, animal_sem_id_2, 
                   animal_sem_id_3, animal_sem_id_4]

# retorna a quantidade de animais que salvei na lista
qtd_sem_id = len(conjunto_sem_id)

# previsao do programa
resultado = modelo.predict(conjunto_sem_id)

# resultado esperado do modelo do programa
resultado_esperado = [gato, gato, cachorro, cachorro]

# calculando a taxa de erros 
# gato = 1, se estiver certo 1-1 =0
taxa_erros = resultado - resultado_esperado

# armazena os acertos
acertos = [d for d in taxa_erros if d == 0]

#retorna a qtd de acertos
qtd_acertos = len(acertos)

print(qtd_acertos)

taxa_acertos = 100 * (qtd_acertos / qtd_sem_id)

print(taxa_acertos)