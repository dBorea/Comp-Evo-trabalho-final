import numpy as np
import random
import matplotlib.pyplot as plt
import math
from typing import Final, List, Optional
from textwrap import wrap

# Definição das constantes de horários
indeterminado: Final = 0
primeiro_seg: Final = 1
segundo_seg: Final = 2
primeiro_ter: Final = 3
segundo_ter: Final = 4
primeiro_qua: Final = 5
segundo_qua: Final = 6
primeiro_qui: Final = 7
segundo_qui: Final = 8
primeiro_sex: Final = 9
segundo_sex: Final = 10


class Horarios:

    def __init__(self,
                 first: int,
                 second: Optional[int] = None,
                 third: Optional[int] = None):
        self.first = first
        self.second = second
        self.third = third

    def get_horarios(self) -> List[int]:
        if self.third is not None and self.second is not None:
            return [self.first, self.second, self.third]
        return [self.first, self.second
                ] if self.second is not None else [self.first]


class Materia:

    def __init__(self, id: str, nome: str, horarios: Horarios,
                 dependencias: List[str], tx_aprovacao: float, periodo: int):
        self.id = id
        self.nome = nome
        self.horarios = horarios
        self.dependencias = dependencias
        self.tx_aprovacao = tx_aprovacao
        self.status = "disponivel"  # disponivel ou concluida
        self.periodo = periodo

    def concluir(self):
        self.status = "concluida"

    def esta_disponivel(self) -> bool:
        return self.status == "disponivel"


class Atividades:

    def __init__(self):
        self.atividades = []

    def adiciona_atividade(self, atividade: Materia):
        self.atividades.append(atividade)

    def get_atividades_disponiveis(self) -> List[Materia]:
        return [
            atividade for atividade in self.atividades
            if atividade.esta_disponivel()
        ]

    def aloca_atividades(self, atividades_semestre):
        for atividade in self.atividades:
            if atividade.id in [a.id for a in atividades_semestre]:
                atividade.concluir()

    def get_atividade_por_id(self, id: str) -> Optional[Materia]:
        for atividade in self.atividades:
            if atividade.id == id:
                return atividade
        return None

    def get_num_horarios_max(self):
        num_horarios = 0
        for atividade in self.atividades:
            if atividade.esta_disponivel():
                num_horarios += sum(x is not None
                                    for x in atividade.horarios.get_horarios())
        return num_horarios


# Definição das matérias e horários
atividades = Atividades()
# 1 SEMESTRE
atividades.adiciona_atividade(
    Materia("DCC203", "PROGRAMAÇÃO E DESENVOLVIMENTO DE SOFTWARE I",
            Horarios(segundo_ter, primeiro_qui), [], 0.492, 1))
atividades.adiciona_atividade(
    Materia("ELE630", "INTRODUCAO A ENGENHARIA DE SISTEMAS",
            Horarios(segundo_sex), [], 0.793, 1))
atividades.adiciona_atividade(
    Materia("MAT001", "CALCULO DIFERENCIAL E INTEGRAL I",
            Horarios(primeiro_seg, segundo_qua, primeiro_sex), [], 0.503, 1))
atividades.adiciona_atividade(
    Materia("MAT038", "GEOMETRIA ANALITICA E ALGEBRA LINEAR",
            Horarios(primeiro_ter, segundo_qui), [], 0.532, 1))
atividades.adiciona_atividade(
    Materia("QUI628", "QUÍMICA GERAL E", Horarios(segundo_seg, primeiro_qua),
            [], 0.214, 1))

# 2 SEMESTRE
atividades.adiciona_atividade(
    Materia("DCC204", "PROGRAMAÇÃO E DESENVOLVIMENTO DE SOFTWARE II",
            Horarios(segundo_ter, primeiro_qui), ["DCC203"], 0.623, 2))
atividades.adiciona_atividade(
    Materia("DCC217", "MATEMATICA DISCRETA PARA ENGENHARIA",
            Horarios(primeiro_ter, segundo_sex), [], 0.580, 2))
atividades.adiciona_atividade(
    Materia("ELT124", "SISTEMAS DIGITAIS", Horarios(segundo_qui, primeiro_sex),
            [], 0.586, 2))
atividades.adiciona_atividade(
    Materia("FIS065", "FUNDAMENTOS DE MECANICA",
            Horarios(segundo_seg, primeiro_qua), [], 0.465, 2))
atividades.adiciona_atividade(
    Materia("MAT039", "CALCULO DIFERENCIAL E INTEGRAL II",
            Horarios(primeiro_seg, segundo_qua), ["MAT001", "MAT038"], 0.514,
            2))

# 3 SEMESTRE
atividades.adiciona_atividade(
    Materia("DCC205", "ESTRUTURA DE DADOS",
            Horarios(segundo_ter, primeiro_qui), ["DCC204"], 0.432, 3))
atividades.adiciona_atividade(
    Materia("ELE064", "ANALISE DE CIRCUITOS ELETRICOS I",
            Horarios(primeiro_sex), ["MAT001"], 0.453, 3))
atividades.adiciona_atividade(
    Materia("ELT029", "LABORATORIO DE SISTEMAS DIGITAIS",
            Horarios(segundo_sex), ["ELT124"], 0.808, 3))
atividades.adiciona_atividade(
    Materia("EST773", "FUNDAMENTOS DE ESTATISTICA E CIÊNCIA DOS DADOS",
            Horarios(primeiro_ter, segundo_qui), ["MAT001"], 0.672, 3))
atividades.adiciona_atividade(
    Materia("FIS069", "FUNDAMENTOS DE ELETROMAGNETISMO",
            Horarios(segundo_seg, primeiro_qua), ["FIS065", "MAT039"], 0.489,
            3))
atividades.adiciona_atividade(
    Materia("MAT040", "EQUACOES DIFERENCIAIS C",
            Horarios(primeiro_seg, segundo_qua), ["MAT039"], 0.781, 3))

# 4 SEMESTRE
atividades.adiciona_atividade(
    Materia("ELE065", "ANALISE DE CIRCUITOS ELETRICOS II",
            Horarios(primeiro_sex), ["ELE064", "MAT040"], 0.548, 4))
atividades.adiciona_atividade(
    Materia("ELE631", "ANÁLISE, PROJETO E PROGRAMAÇÃO ORIENTADOS A OBJETOS",
            Horarios(primeiro_ter, segundo_qui), ["DCC204"], 0.731, 4))
atividades.adiciona_atividade(
    Materia("ELE632", "PROCESSOS E MÉTODOS EM ENGENHARIA DE SISTEMAS",
            Horarios(segundo_ter, primeiro_qui), [], 0.750, 4))
atividades.adiciona_atividade(
    Materia("ESA019", "CIÊNCIAS DO AMBIENTE", Horarios(segundo_sex), [], 0.750,
            4))
atividades.adiciona_atividade(
    Materia("FIS086", "FUNDAMENTOS DE OSCILACOES, ONDAS E OPTICA",
            Horarios(segundo_seg, primeiro_qua), ["FIS069"], 0.752, 4))
atividades.adiciona_atividade(
    Materia("MAT002", "CALCULO DIFERENCIAL E INTEGRAL III",
            Horarios(primeiro_seg, segundo_qua), ["MAT039"], 0.473, 4))

# 5 SEMESTRE
atividades.adiciona_atividade(
    Materia("EEE048", "FUNDAMENTOS DE INTELIGÊNCIA ARTIFICIAL",
            Horarios(segundo_ter, primeiro_qui), [], 0.825, 5))
atividades.adiciona_atividade(
    Materia("ELE028", "LABORATÓRIO DE CIRCUITOS ELÉTRICOS I",
            Horarios(primeiro_sex), ["ELE064", "FIS069"], 0.804, 5))
atividades.adiciona_atividade(
    Materia("ELT136", "FUNDAMENTOS DE SISTEMAS DINÂMICOS E CONTROLE",
            Horarios(primeiro_ter, segundo_qui), ["MAT040"], 0.599, 5))
atividades.adiciona_atividade(
    Materia("EMT122", "INTRODUÇÃO À CIÊNCIA DOS MATERIAIS",
            Horarios(segundo_seg), [], 0.921, 5))
atividades.adiciona_atividade(
    Materia("ELE082", "PESQUISA OPERACIONAL",
            Horarios(primeiro_qua, segundo_qua), ["MAT040"], 0.864, 5))

# 6 SEMESTRE
atividades.adiciona_atividade(
    Materia("DCC011", "INTRODUÇÃO A BANCO DE DADOS",
            Horarios(primeiro_ter, primeiro_qui), ["DCC204"], 0.694, 6))
atividades.adiciona_atividade(
    Materia("ELE077", "OTIMIZACAO NAO LINEAR", Horarios(segundo_qui),
            ["MAT040"], 0.644, 6))
atividades.adiciona_atividade(
    Materia("ELT084", "DISPOSIT.E CIRCUITOS ELETRONICOS BASICOS",
            Horarios(segundo_seg, primeiro_qua), ["ELE065"], 0.668, 6))
atividades.adiciona_atividade(
    Materia("ELT123", "ARQUITETURA E ORGANIZAÇÃO DE COMPUTADORES",
            Horarios(primeiro_seg, segundo_qua), ["ELT124"], 0.600, 6))
atividades.adiciona_atividade(
    Materia("FIS152", "FUNDAMENTOS DE MECÂNICA DOS FLUIDOS E TERMODINÂMICA",
            Horarios(segundo_ter), ["FIS065", "MAT001"], 0.697, 6))

# 7 SEMESTRE
atividades.adiciona_atividade(
    Materia("EEE046", "SISTEMAS A EVENTOS DISCRETOS",
            Horarios(segundo_ter, primeiro_qui), ["ELT124"], 0.557, 7))
atividades.adiciona_atividade(
    Materia("EEE049", "APRENDIZADO DE MÁQUINA",
            Horarios(primeiro_ter, segundo_qui), [], 0.748, 7))
atividades.adiciona_atividade(
    Materia("EEE050", "MODELAGEM E SIMULAÇÃO MULTIFÍSICA",
            Horarios(primeiro_seg, segundo_seg),
            ["FIS152", "FIS069", "MAT040"], 0.709, 7))
atividades.adiciona_atividade(
    Materia("ELE088", "TEORIA DA DECISAO", Horarios(segundo_sex), [], 0.912,
            7))
atividades.adiciona_atividade(
    Materia("ELE633", "LABORATORIO DE SISTEMAS I",
            Horarios(primeiro_qua, segundo_qua), ["ELE631"], 0.955, 7))
atividades.adiciona_atividade(
    Materia("ELT080", "LABORATORIO DE CIRCUITOS ELETRONICOS E PROJETOS",
            Horarios(primeiro_sex), ["ELT084"], 0.937, 7))

# 8 SEMESTRE
atividades.adiciona_atividade(
    Materia("DCC218", "INTRODUÇÃO A SISTEMAS COMPUTACIONAIS",
            Horarios(primeiro_seg, segundo_qua), ["DCC204"], 0.717, 8))
atividades.adiciona_atividade(
    Materia("EEE017", "CONFIABILIDADE DE SISTEMAS",
            Horarios(segundo_seg, primeiro_qua), ["EST773"], 0.914, 8))
atividades.adiciona_atividade(
    Materia("ELE634", "LABORATORIO DE SISTEMAS II",
            Horarios(primeiro_sex, segundo_sex), ["ELE633"], 0.957, 8))

# 9 SEMESTRE
atividades.adiciona_atividade(
    Materia("ELE635", "LABORATORIO DE SISTEMAS III",
            Horarios(primeiro_ter, segundo_ter),
            ["EEE046", "ELT029", "ELT123", "ELT080", "ELE634"], 0.964, 9))
atividades.adiciona_atividade(
    Materia("EEE051", "LABORATORIO DE GERENCIAMENTO DE SISTEMAS",
            Horarios(primeiro_qui, segundo_qui), [], 0.985, 9))

# # 10 SEMESTRE
# atividades.adiciona_atividade(
#     Materia("EEEXXP", "TCC I", Horarios(indeterminado), [], 0.789))

# # 11 SEMESTRE
# atividades.adiciona_atividade(
#     Materia("EEEXXQ", "TCC II", Horarios(indeterminado), [], 0.777))

# 12 SEMESTRE
# atividades.adiciona_atividade(
#     Materia("EEE054", "ESTÁGIO", Horarios(primeiro_sex), [], 0.928))


def criar_individuo(tamanho_genoma):
    return np.random.choice(a=[0, 1], size=(tamanho_genoma, ))


def criar_populacao(tamanho_pop, tamanho_genoma):
    return [criar_individuo(tamanho_genoma) for _ in range(tamanho_pop)]


def num_conflitos(individuo, atividades, materia):
    conflitos = 0
    materia_index = atividades.index(
        materia)  # Encontra o índice da matéria na lista de atividades
    if individuo[
            materia_index] == 1:  # Verifica se a matéria está no cronograma
        for i, mat in enumerate(atividades):
            if individuo[i] == 1 and mat.id != materia.id:
                for horario in materia.horarios.get_horarios():
                    for outro_horario in mat.horarios.get_horarios():
                        if horario == outro_horario:
                            conflitos += 1
    return conflitos


def num_dependencias(individuo, atividades, materia):
    dependencias_encontradas = 0
    for dependencia_id in (materia.dependencias):
        for outra_materia in (atividades):
            # print(dependencia_id, outra_materia.id)
            if (outra_materia.id == dependencia_id):
                dependencias_encontradas += 1
    # print(materia.nome, dependencias_encontradas)
    return dependencias_encontradas


def min_periodo_pendente(atividades):
    min_periodo = 100
    for atv in atividades:
        if (atv.periodo < min_periodo):
            min_periodo = atv.periodo
    return min_periodo


def avaliar_individuo(individuo, atividades, semestre_atual, n_horarios_max):
    fit = 0
    num_horarios_total = 0
    conflitos = 0
    dependencias = 0
    min_periodo = 100
    max_periodo = 0

    for i, gene in enumerate(individuo):
        # print(gene)
        if gene:
            materia = atividades[i]
            if (materia.periodo < min_periodo):
                min_periodo = materia.periodo
            if (materia.periodo > max_periodo):
                max_periodo = materia.periodo
            # print(i, materia.nome)
            dependencias += num_dependencias(individuo, atividades, materia)
            conflitos += num_conflitos(individuo, atividades, materia)

            horarios_materia = materia.horarios.get_horarios()
            num_horarios_total += len(horarios_materia)
            fit += (len(horarios_materia) * 300 * materia.tx_aprovacao /
                    max(1, max_periodo - min_periodo_pendente(atividades))**2)

    pen_hor_sobrando = min(10, n_horarios_max) - num_horarios_total
    if (pen_hor_sobrando < 0):
        pen_hor_sobrando = abs(pen_hor_sobrando) * 10

    fit = fit - 100 * pen_hor_sobrando

    periodo_dif = abs(max_periodo - min_periodo)
    fit = -abs(fit) - 100 * periodo_dif if periodo_dif > 2 else fit

    pendencia_dif = semestre_atual - min_periodo_pendente(atividades)
    fit = -abs(fit) - 100 * pendencia_dif if pendencia_dif > 2 else fit

    fit = -abs(fit) * (dependencias + conflitos) * 100 if (
        conflitos > 0 or dependencias > 0) else fit

    # print(f"individuo: {individuo}")
    # print(f"conflito: {conflitos}, dependencia:{dependencias}")
    # print(f"print individuo: {fit}")
    return fit


def selecao_torneio(populacao, fitness_pop, tamanho_torneio):
    nova_populacao = []
    while len(nova_populacao) < len(populacao):
        torneio_indices = random.sample(range(len(populacao)), tamanho_torneio)
        torneio_participantes = [(populacao[i], fitness_pop[i])
                                 for i in torneio_indices]
        vencedor = max(torneio_participantes, key=lambda x: x[1])
        nova_populacao.append(vencedor[0])
    return nova_populacao


def crossover(pai1, pai2):
    if len(pai1) > 2:
        ponto_corte = np.random.randint(1, len(pai1) - 1)
    else:
        return pai1
    filho = np.concatenate([pai1[:ponto_corte], pai2[ponto_corte:]])

    return filho


def mutacao(individuo, taxa_mutacao):
    for i in range(len(individuo)):
        if np.random.random() < taxa_mutacao:
            individuo[i] = 1 if individuo[i] == 0 else 0
    return individuo


# fit_ideal = 30 * 10 * 0.8


def algoritmo_evolutivo(tamanho_pop, tamanho_genoma, limite_geracoes,
                        taxa_mutacao, tamanho_torneio, semestre_atual,
                        atividades, n_horarios_max):
    populacao = criar_populacao(tamanho_pop, tamanho_genoma)
    top_fit = -10000000000000000000000
    gen_found = 0
    top_idv = []

    for gen in range(limite_geracoes):

        fitness_pop = [
            avaliar_individuo(individuo, atividades, semestre_atual,
                              n_horarios_max) for individuo in populacao
        ]

        vencedores_torneio = selecao_torneio(populacao, fitness_pop,
                                             tamanho_torneio)
        nova_populacao = []

        while len(nova_populacao) < tamanho_pop:
            pai1, pai2 = random.sample(vencedores_torneio, 2)
            filho1 = crossover(pai1, pai2)
            # filho2 = crossover(pai2, pai1)
            nova_populacao.extend([filho1])

        # nova_populacao = nova_populacao[:tamanho_pop]
        populacao = [
            mutacao(individuo, taxa_mutacao) for individuo in nova_populacao
        ]

        melhor_individuo = max(
            populacao,
            key=lambda x: avaliar_individuo(x, atividades, semestre_atual,
                                            n_horarios_max))

        melhor_fitness = avaliar_individuo(melhor_individuo, atividades,
                                           semestre_atual, n_horarios_max)

        # print(melhor_fitness)

        if top_fit < melhor_fitness:
            top_fit = melhor_fitness
            gen_found = gen
            top_idv = melhor_individuo.copy()

        # if gen % 10 == 0:
        #     print(f"Geração {gen}, melhor fitness {melhor_fitness}")

        # if top_fit >= fit_ideal:
        #     break

    return top_idv, top_fit, gen_found


curriculo_escolar = []
semestre = [
    atividades.get_atividade_por_id("MAT001"),
    atividades.get_atividade_por_id("QUI628"),
    atividades.get_atividade_por_id("MAT038"),
    atividades.get_atividade_por_id("DCC203"),
    atividades.get_atividade_por_id("ELE630"),
]

atividades.aloca_atividades(semestre)
curriculo_escolar.append(semestre)

atividades_disponiveis = atividades.get_atividades_disponiveis()
qtd_atividades_disponiveis = len(atividades_disponiveis)

semestre_atual = 1
while (qtd_atividades_disponiveis > 1):
    semestre = []
    semestre_atual += 1

    solucao, melhor_fitness, geracao_encontrado = algoritmo_evolutivo(
        # tamanho_pop=4 * qtd_atividades_disponiveis,
        tamanho_pop=100,
        tamanho_genoma=qtd_atividades_disponiveis,
        # limite_geracoes=20 * qtd_atividades_disponiveis,
        limite_geracoes=500,
        taxa_mutacao=0.05,
        tamanho_torneio=3,
        semestre_atual=semestre_atual,
        atividades=atividades_disponiveis,
        n_horarios_max=atividades.get_num_horarios_max())

    # Resultados
    print(f"Semestre {semestre_atual}")
    print(f"Melhor solução encontrada: {solucao}")
    print(f"Fitness: {melhor_fitness}, na geração {geracao_encontrado}")

    avaliar_individuo(solucao, atividades_disponiveis, semestre_atual,
                      atividades.get_num_horarios_max())

    # Seleciona as matérias da solução
    for i, gene in enumerate(solucao):
        if gene == 1:
            semestre.append(atividades_disponiveis[i])

    atividades.aloca_atividades(semestre)
    curriculo_escolar.append(semestre)

    atividades_disponiveis = atividades.get_atividades_disponiveis()
    qtd_atividades_disponiveis = len(atividades_disponiveis)

# verifica se sobrou materia
atividades_disponiveis = atividades.get_atividades_disponiveis()
qtd_atividades_disponiveis = len(atividades_disponiveis)
if (qtd_atividades_disponiveis > 0):
    atividades.aloca_atividades(atividades_disponiveis)
    curriculo_escolar.append(atividades_disponiveis)


def imprime_curriculo(curriculo_escolar, exec_folder):
    print_file = open(f"{exec_folder}/curriculo.txt", "w")
    aprv_curriculo = 0
    for i, semestre in enumerate(curriculo_escolar):
        print_file.write(f"{i+1}º SEMESTRE\n")
        aprv_semestre = 0
        for j, materia in enumerate(semestre):
            n_hor = sum(x is not None for x in materia.horarios.get_horarios())

            print_file.write(
                f"Periodo {materia.periodo} - {materia.id}: {materia.nome}\n")
            aprv_semestre += materia.tx_aprovacao
        print_file.write(
            f"Média de aprovação do semestre: {aprv_semestre/(j+1):.2%}\n")
        aprv_curriculo += aprv_semestre / (j + 1)
    print_file.write(
        f"\nMédia de aprovação do currículo: {aprv_curriculo/(i+1):.2%}\n")


#PLOTS
def plot_curriculo(curriculo_escolar, exec_folder):
    for i, semestre in enumerate(curriculo_escolar):
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 5)
        cmap = iter([plt.cm.tab20(i) for i in range(20)])
        labels = []

        for materia in semestre:
            randcolor = next(cmap)
            for horario in materia.horarios.get_horarios():
                if (horario is not None):
                    x_coord = math.ceil(horario / 2)
                    height = 5 / 3
                    bottom = (horario % 2 == 0) * (5.5 / 3) + 19

                    ax.bar(x_coord,
                           height=height,
                           bottom=bottom,
                           color=randcolor)
                    labels.append(f"{materia.nome}")

        x_labels = ["Seg", "Ter", "Qua", "Qui", "Sex"]
        labels = ['\n'.join(wrap(l, 15)) for l in labels]
        [
            ax.bar_label(container, fmt=labels[i], label_type="center")
            for i, container in enumerate(ax.containers)
        ]
        plt.xticks([1, 2, 3, 4, 5], x_labels)
        plt.yticks([19, 19.5, 20, 20.5, 21, 21.5, 22, 22.5],
                   labels=[
                       "19:00", "19:30", "20:00", "20:30", "21:00", "21:30",
                       "22:00", "22:30"
                   ])

        plt.gca().invert_yaxis()
        plt.savefig(f'{exec_folder}/Semestre {i+1}')


exec_folder = "Exec test"
imprime_curriculo(curriculo_escolar, exec_folder)
plot_curriculo(curriculo_escolar, exec_folder)
