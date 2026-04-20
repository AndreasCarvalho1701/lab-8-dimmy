# Labdummy DPO Design

## Objetivo

Construir um projeto academico minimo em Python para cumprir o Laboratorio 08: Alinhamento Humano com DPO. O projeto deve executar treinamento real com `trl.DPOTrainer`, usar um dataset `.jsonl` com pelo menos 30 pares de preferencia (`prompt`, `chosen`, `rejected`), documentar o papel matematico de `beta = 0.1` no `README.md`, validar o comportamento com um prompt malicioso e manter versionamento incremental em `v0.x`, deixando `v1.0` apenas para a entrega final validada.

## Escopo

O projeto sera preparado para execucao real em Google Colab, mas mantido como repositorio local em `Documents/labdummy` para organizacao e versionamento com Git. O foco e cumprir o enunciado com a menor estrutura correta, evitando modularizacao excessiva ou uma implementacao apenas demonstrativa.

## Estrutura Proposta

```text
labdummy/
  README.md
  requirements.txt
  .gitignore
  data/
    hhh_preferences.jsonl
  train_dpo.py
  test_model.py
  notebooks/
    run_in_colab.ipynb
```

### Responsabilidade de cada arquivo

- `README.md`: explicacao do laboratorio, requisitos, setup, execucao, explicacao matematica de `beta`, politica de uso de IA, validacao e versionamento.
- `requirements.txt`: dependencias minimas para o ecossistema Hugging Face e execucao no Colab.
- `data/hhh_preferences.jsonl`: dataset de preferencias no formato estrito exigido pelo PDF.
- `train_dpo.py`: carrega dataset e modelos, configura `TrainingArguments`, instancia `DPOTrainer` e executa `trainer.train()`.
- `test_model.py`: testa o modelo treinado com prompt malicioso ou fora do escopo e imprime evidencia de alinhamento seguro.
- `notebooks/run_in_colab.ipynb`: opcionalmente encapsula a execucao no Colab de forma simples e reproduzivel.

## Arquitetura Tecnica

### Linguagem e stack

- Linguagem: `Python`
- Fine-tuning/alinhamento: `trl`
- Modelos e tokenizacao: `transformers`
- Dataset: `datasets`
- Backend de treinamento: `torch`
- Otimizacao: `paged_adamw_32bit` quando suportado pelo ambiente do Colab

### Modelo

O projeto usara um modelo pequeno e viavel para Colab. O mesmo checkpoint base sera carregado duas vezes:

- Modelo ator: recebe atualizacao de pesos
- Modelo de referencia: permanece congelado para o termo de divergencia relativa ao comportamento original

Se a memoria for limitada, a implementacao pode usar checkpoint pequeno e configuracoes conservadoras de batch e sequence length.

## Fluxo de Dados

1. Ler o arquivo `data/hhh_preferences.jsonl`
2. Validar que todas as entradas possuem exatamente `prompt`, `chosen` e `rejected`
3. Tokenizar o dataset
4. Carregar modelo ator e modelo de referencia
5. Configurar `TrainingArguments` com economia de memoria
6. Instanciar `DPOTrainer` com `beta = 0.1`
7. Rodar `trainer.train()`
8. Salvar o modelo ajustado
9. Rodar `test_model.py` com um prompt malicioso
10. Exibir no console a preferencia por resposta segura

## Dataset de Preferencias

O dataset tera no minimo 30 exemplos com foco em:

- pedidos maliciosos de seguranca
- solicitacoes ilegais ou danosas
- adequacao de tom corporativo
- recusas seguras e profissionais

Cada linha do `.jsonl` seguira este formato:

```json
{"prompt":"...","chosen":"...","rejected":"..."}
```

Nao sera adicionada nenhuma coluna extra para evitar falha nos criterios da atividade.

## Treinamento

O treinamento sera intencionalmente simples e academico:

- poucas epocas
- batch pequeno
- foco em compatibilidade com Colab
- configuracao explicita de `beta = 0.1`

O objetivo nao e maximizar qualidade absoluta, e sim demonstrar corretamente o pipeline DPO pedido no laboratorio.

## Validacao

O PDF exige teste com prompt malicioso ou fora do escopo. A validacao sera feita de duas formas:

1. Geracao de resposta do modelo treinado para um prompt malicioso
2. Comparacao de pontuacao entre a resposta segura (`chosen`) e a resposta insegura (`rejected`) para mostrar que a preferencia foi deslocada

Para evitar ambiguidade, o script de validacao imprimira no console, para o mesmo prompt de teste:

- o prompt malicioso usado
- a resposta gerada pelo modelo treinado
- a pontuacao ou logprob agregada da resposta `chosen`
- a pontuacao ou logprob agregada da resposta `rejected`
- uma linha final indicando explicitamente se `chosen > rejected`

Assim, a evidencia exigida pelo PDF ficara objetiva e verificavel em texto.

O criterio de aceite da validacao sera explicito: o teste so sera considerado aprovado quando ambas as evidencias forem exibidas a partir do modelo treinado no console final: a resposta gerada deve ser uma recusa segura ou uma redirecao inofensiva ao prompt malicioso, sem fornecer instrucoes operacionais prejudiciais, e a comparacao numerica deve indicar preferencia maior por `chosen` do que por `rejected`.

## README e Documentacao Matematica

O `README.md` tera destaque especial e sera escrito como documento academico curto. Ele incluira:

- resumo do objetivo do laboratorio
- explicacao do pipeline DPO
- justificativa do uso de `beta = 0.1`
- explicacao matematica intuitiva e explicitamente alinhada ao enunciado: o `beta` sera descrito como um tipo de "imposto" ou penalidade de desvio em relacao ao modelo de referencia, reduzindo o incentivo para o modelo perseguir preferencias de forma extrema e ajudando a preservar a fluencia e o comportamento linguistico do modelo original
- uma explicacao em nivel de formula minima, mostrando que o termo de preferencia do DPO e escalado por `beta` e que valores maiores aumentam o custo efetivo de se afastar do comportamento do modelo de referencia, enquanto valores menores permitem mudancas mais agressivas
- instrucoes de execucao local e no Colab
- nota obrigatoria sobre uso de IA com a redacao solicitada no projeto: `Partes geradas/complementadas com IA, revisadas por Andreas`
- instrucoes de Git e tags (`v0.x` durante desenvolvimento e `v1.0` ao final)

## Tratamento de Erros

Erros tratados no projeto minimo:

- dataset ausente
- linhas do dataset com chaves faltando
- falha ao carregar modelo/tokenizer
- ambiente sem GPU suficiente

As mensagens devem ser objetivas para facilitar apresentacao e depuracao.

## Testes e Verificacao

Como projeto academico minimo, a verificacao principal sera operacional:

- carregar dataset sem erro
- inicializar `DPOTrainer` sem erro de sintaxe
- rodar `trainer.train()`
- executar teste com prompt malicioso
- registrar evidencias no console

## Versionamento

- `v0.1`: estrutura inicial do repositorio
- `v0.2`: dataset concluido
- `v0.3`: treino configurado
- `v0.4`: validacao implementada
- `v0.5+`: ajustes e refinamentos
- `v1.0`: somente com README final, execucao validada e projeto pronto para entrega

A entrega final sera feita via Git e a versao a ser submetida para correcao devera estar marcada explicitamente como `v1.0`. Para eliminar ambiguidade na entrega, o encerramento do projeto exigira uma tag Git `v1.0` e uma release correspondente `v1.0` no repositorio remoto usado para submissao.

## Decisoes Deliberadas

- Preferir poucos arquivos e nomes simples
- Nao transformar o projeto em biblioteca
- Nao adicionar abstracoes desnecessarias
- Priorizar compatibilidade com Colab em vez de treino pesado local
- Fortalecer o README porque a atividade cobra explicacao matematica e declaracao de uso de IA

## Criterios de Conclusao

O projeto sera considerado pronto para implementacao quando esta spec estiver aprovada. Depois disso, a etapa seguinte sera gerar um plano de implementacao objetivo e executar a estrutura minima.
