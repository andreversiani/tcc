# Caracterização e Extração de Atributos de Faltas de Alta Impedância em Vegetação: Aplicação em Dados Reais

**Trabalho de Conclusão de Curso — Engenharia de Sistemas, UFMG**
**Autor:** André Versiani de Mattos
**Orientadora:** Prof.ª Dra. Gabriela Nunes Lopes
**Previsão de apresentação:** Dezembro de 2026

---

## Sobre o Projeto

Este TCC investiga Faltas de Alta Impedância em vegetação (VeHIFs — *Vegetation High-Impedance Faults*) em sistemas de distribuição elétrica, com foco em extração de atributos de sinal e classificação supervisionada usando dados reais de campo.

O problema central: quando um condutor energizado entra em contato com vegetação (galhos, folhas, árvores), a corrente de falta gerada é tão baixa que não aciona proteções convencionais de sobrecorrente. Mesmo assim, essa corrente pode carbonizar a matéria vegetal e gerar brasas incandescentes, representando risco real de incêndio. O incidente motivador foi o "Black Saturday" (Vitória, Austrália, fevereiro de 2009): 173 mortes, mais de 2.000 residências destruídas, com falhas em ativos elétricos apontadas como causa contribuinte.

**Objetivo geral:** Caracterizar e extrair atributos de sinais de faltas de alta impedância em vegetação a partir de dados reais, avaliando sua capacidade discriminativa em um problema de classificação supervisionada.

---

## Dados

**Dataset:** VeHIF (*Vegetation Conduction Ignition Test Program*)
**Origem:** Powerline Bushfire Safety Program (PBSP), Governo de Vitória, Austrália
**Contexto:** Rede de distribuição real a 22 kV, trifásica, três fios

Características do dataset:
- Mais de 900 registros de faltas envolvendo vegetação
- Duas classes de canais de medição:
  - **Canal LF (Baixa Frequência):** componente fundamental da forma de onda
  - **Canal HF (Alta Frequência):** componentes espectrais do arco elétrico (2–10 kHz)
- Diversas espécies vegetais nativas australianas
- Variação de umidade e configurações de contato
- Condições reais de campo (ruído operacional, variações de rede)
- Formato original `.pnrf`, disponibilizado publicamente em HDF5 (leitura via Python, MATLAB, C++)

---

## Metodologia

O trabalho é inteiramente implementado em **Python**. As etapas principais são:

1. **Revisão bibliográfica** — FAIs, VeHIFs, processamento de sinais, algoritmos de classificação
2. **Análise exploratória do dataset** — inspeção dos registros, definição formal das classes (falta vs. operação normal), análise estatística descritiva dos sinais
3. **Pré-processamento** — filtragem, normalização, segmentação por janelas temporais, tratamento de valores ausentes ou corrompidos
4. **Extração de atributos** — domínio do tempo, frequência e tempo-frequência:
   - Tempo: energia, assimetria, curtose, *buildup*, rugosidade
   - Frequência: componentes harmônicas, inter-harmônicas, conteúdo de alta frequência
   - Tempo-frequência: Transformada Wavelet Discreta (DWT) e STFT
5. **Seleção de atributos** — análise de importância, discriminação estatística
6. **Treinamento e avaliação de classificadores** — Árvore de Decisão, SVM, XGBoost, Redes Neurais Artificiais, com validação cruzada estratificada (métricas: Acurácia, Precisão, Recall, F1-score)
7. **Análise dos resultados** — atributos mais discriminativos, relação com características elétricas das VeHIFs

---

## Características Técnicas das FAIs (contexto teórico)

Aspectos importantes para entender os atributos extraídos:

- **Não-linearidade:** o comportamento do arco elétrico é não-linear e variante no tempo, não uma resistência constante
- **Assimetria:** corrente assimétrica entre semiciclos positivo e negativo
- **Efeito buildup:** corrente cresce progressivamente nos primeiros ciclos (aquecimento e evolução do contato)
- **Alta frequência:** componentes entre 2–10 kHz são assinaturas chave das FAIs; ausentes em operação normal

---

## Stack Tecnológica

| Componente | Tecnologia |
|---|---|
| Linguagem principal | Python |
| Processamento de sinais | NumPy, SciPy, PyWavelets |
| Machine Learning | scikit-learn, XGBoost |
| Leitura de dados | HDF5 (h5py) |
| Documento acadêmico | LaTeX (ABNT), compilado com pdflatex + bibtex |
| Build | Makefile |
| IDE | VS Code (LaTeX Workshop + Python) |
| Controle de versão | Git |

---

## Estrutura do Repositório

```
TCC/
├── README.md
├── monografia/              # Documento LaTeX (tese)
│   ├── main.tex             # Arquivo mestre
│   ├── estilo.cls           # Classe ABNT customizada
│   ├── referencias.bib      # 31 referências bibliográficas
│   ├── main.pdf             # PDF compilado
│   ├── makefile
│   ├── 01-elementos-pre-textuais/
│   ├── 02-elementos-textuais/
│   │   ├── introducao.tex          ✅ Completo
│   │   ├── revisaobibliografica.tex ✅ Completo
│   │   ├── metodologia.tex          🔲 Esboço (seções definidas, sem conteúdo)
│   │   └── visão-geral.tex          ✅ Completo
│   └── 03-elementos-pos-textuais/
└── codes/                   # Implementação Python (a desenvolver)
```

---

## Estado Atual de Desenvolvimento

**Escrita da monografia:**
- ✅ Introdução — problema, motivação, objetivos
- ✅ Revisão bibliográfica — FAIs, métodos de detecção, técnicas de processamento (DFT, STFT, DWT), algoritmos de ML, dataset VeHIF
- ✅ Visão geral — motivação, objetivos, escopo, metodologia resumida
- 🔲 Metodologia — seções definidas, aguardando conteúdo detalhado
- 🔲 Resultados e discussão — não iniciado
- 🔲 Conclusões — não iniciado
- 🔲 Resumos (PT/EN) — placeholders

**Implementação:**
- 🔲 Toda a pasta `codes/` ainda não foi iniciada

---

## Escopo

**Incluído:**
- Caracterização e classificação automática de VeHIFs com o dataset público VeHIF
- Pré-processamento, extração de atributos (tempo, frequência, tempo-frequência) e seleção
- Treinamento e comparação de quatro algoritmos de classificação
- Validação e análise em dados reais

**Excluído:**
- Novos modelos de simulação de FAIs
- Sistemas de proteção em tempo real ou relés
- Coleta de novos dados de campo
- FAIs em superfícies não-vegetais
- Localização de faltas na rede

---

## Referências-Chave

A bibliografia (`referencias.bib`) contém 31 referências, incluindo:

- Aucoin & Russell (1982) — trabalho fundacional sobre detecção por alta frequência
- Gomes (2018) — metodologia de extração de atributos para FAIs
- Ozansoy et al. (2020) — descrição do dataset VeHIF
- Yang et al. (2025) — FAIs em vegetação com dados reais, XGBoost alcançando 98,17% de acurácia
- Relatórios do PBSP — contexto do programa australiano de segurança contra incêndios

---

## Contexto para Prompts Futuros

Ao assistir o desenvolvimento deste projeto, considere:

1. **Audiência:** trabalho de graduação em Engenharia de Sistemas na UFMG, com nível técnico avançado em ML e processamento de sinais
2. **Norma:** documento LaTeX no padrão ABNT (arquivo `estilo.cls`); citações no formato `\cite{}` do BibTeX
3. **Idioma da monografia:** português brasileiro (exceto termos técnicos consagrados em inglês)
4. **Idioma do código:** inglês (variáveis, funções, comentários)
5. **Dataset VeHIF:** dados reais, não simulados — isso é um diferencial explícito do trabalho
6. **Foco da contribuição:** avaliação sistemática e comparativa de atributos × classificadores em dados reais; não há consenso na literatura sobre quais combinações funcionam melhor com o VeHIF
7. **Próxima etapa prioritária:** completar o capítulo de Metodologia e iniciar a implementação em Python na pasta `codes/`
