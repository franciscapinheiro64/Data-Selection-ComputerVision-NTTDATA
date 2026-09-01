# smart-embedding-clustering-cv

Seleção Inteligente de Dados por Embeddings para Visão Computacional

Repositório do estágio de verão — agrupamento de imagens semelhantes via embeddings (DINOv2/CLIP) e clustering clássico (k-means).

---

## Links essenciais

- **Plano de estágio completo** (objetivos, semana a semana, critérios de sucesso, riscos): [Plano no Loop](https://loop.cloud.microsoft/p/eyJ1IjoiaHR0cHM6Ly9ldmVyaXNncm91cC5zaGFyZXBvaW50LmNvbS9jb250ZW50c3RvcmFnZS9DU1BfYTVlYjBiNzAtZjY2Zi00YjA3LWIzODYtZGZkN2ZjYTdhNmFhP25hdj1jejBsTWtaamIyNTBaVzUwYzNSdmNtRm5aU1V5UmtOVFVGOWhOV1ZpTUdJM01DMW1OalptTFRSaU1EY3RZak00Tmkxa1ptUTNabU5oTjJFMllXRW1aRDFpSlRJeFkwRjJjbkJYWHpKQ01IVjZhSFJmV0Y5TFpXMXhkVFphTnkxWlMwSkZkRWRwYkZGcmVVZGFhRTg1WjNkQlFuQkNlREpKWlZKS1FtcFdSMms0YmtaVFl5Wm1QVEF4TmxwU01sWlVOMGhLVUVSQlJWbFlVVmRTUkRNM1QxSkZNa0V5TmxFMFdETW1ZejBsTWtZbVlUMU1iMjl3UVhCd0puQTlKVFF3Wm14MWFXUjRKVEpHYkc5dmNDMXdZV2RsTFdOdmJuUmhhVzVsY2laNFBTVTNRaVV5TW5jbE1qSWxNMEVsTWpKVU1GSlVWVWg0YkdSdFZubGhXRTV1WTIwNU1XTkROWHBoUjBaNVdsaENkbUZYTlRCTWJVNTJZbGg0YVVsWFRrSmtia3AzVmpFNGVWRnFRakZsYldnd1dERm9abE15Vm5SaldGVXlWMnBqZEZkVmRFTlNXRkpJWVZkNFVtRXpiRWhYYldoUVQxZGtNMUZWU25kUmJtZDVVMWRXVTFOclNuRldhMlJ3VDBjMVIxVXlUamhOUkVVeVYyeEplVlpzVWxwTk1XaFhUbXN4VkZSNlZYcFRhbFpEVFRCMFdFNUZUbE5VYWtwQ1RsVnpNVlYzSlRORUpUTkVKVEl5SlRKREpUSXlhU1V5TWlVelFTVXlNalk0TlRVek1XVmpMVEJqTWpVdE5ERmtZUzA1TTJJM0xXWXdNMkl3WlRrellqWmhaaVV5TWlVM1JBPT0ifQ%3D%3D?ct=1782990137077&&LOF=1)
- **uv cheat sheet** (comandos essenciais): `docs/uv_cheat_sheet.md` (incluído neste repositório)

---

## Objetivo do projeto

Construir e validar um método que agrupa imagens semelhantes a partir dos seus embeddings, entregue como componente reutilizável (`GroupingStrategy`). Dataset de trabalho: frutas (Kaggle, sshikamaru/fruit-recognition). Detalhe completo no Plano (link acima).

## Objetivos de aprendizagem

- Como uma imagem se torna dados que um algoritmo manipula.
- O que é um embedding e um espaço vetorial de similaridade.
- Clustering (k-means) e como avaliar a qualidade de grupos.
- Autonomia com a stack de CV moderna baseada em modelos pré-treinados.

---

## Setup rápido

```bash
# instalar uv (se ainda não tiver)
curl -LsSf https://astral.sh/uv/install.sh | sh

# clonar o repositório
git clone <url-do-repo>
cd smart-embedding-clustering-cv

# instalar dependências e criar o .venv
uv sync --all-groups

# correr o script inicial dentro do ambiente
uv run python src/hello_world.py
```

Ver `docs/uv_cheat_sheet.md` para todos os comandos do dia a dia (adicionar dependências, correr testes, gerir versão do Python).

---

## Estrutura do repositório

```
smart-embedding-clustering-cv/
├── src/                  # código do projeto (extração de embeddings, clustering, componente reutilizável)
├── notebooks/            # notebooks de desenvolvimento e exploração
├── tests/                # testes unitários (pytest)
├── data/                 # dados locais (fora do controlo de versão)
├── docs/                 # documentação, incluindo uv_cheat_sheet.md
├── pyproject.toml        # metadados e dependências
├── uv.lock               # versões trancadas das dependências (commitar)
└── README.md             # este ficheiro
```

---

## Credenciais e acessos necessários no kickoff

- **Kaggle API**: necessária para download do dataset de frutas (`sshikamaru/fruit-recognition`). Confirmar com os orientadores internos se já está configurada.
- **Acesso a GPU** (se disponível): confirmar antes de comprometer o calendário — em CPU a extração de embeddings no dataset de frutas é viável mas mais lenta.
- **Acesso ao repositório Git**: confirmar permissões de push antes do primeiro dia.

---

## Cadência

- **Standup diário** com os orientadores internos — curto, foco em bloqueios.
- **Demo semanal** ao orientador sénior — mostrar o que funcionou, o que não funcionou, e porquê.
- **Segunda-feira de manhã**: alinhamento da semana (tarefas + dias de presença necessária). Tarde de segunda: Talent/Business Sessions.
- **Sexta-feira à tarde**: livre (horário de verão).

## Papéis

- **Orientador sénior (CV)**: direção técnica, validação da metodologia, revisão semanal.
- **Orientadores internos (engenharia)**: apoio diário, pair programming, qualidade de código.
- **Estagiária**: execução, aprendizagem, comunicação clara de progresso e bloqueios.

---

## Primeira semana — o que esperar

- Download do dataset de frutas.
- Perceber imagem como tensor (canais, resolução, pré-processamento).
- Gerar embeddings de todo o dataset com CLIP (primeira escolha por facilidade de visualização).
- Conseguir explicar por palavras próprias por que imagens semelhantes ficam próximas no espaço vetorial.
- **Marco:** ficheiro de embeddings cobrindo 100% das imagens, com metadados associados.

---

## Recursos de aprendizagem sugeridos

- **Visão geral de CV**: primeiros módulos do CS231n (Stanford) ou fast.ai.
- **Embeddings e modelos fundacionais**: documentação Hugging Face (DINOv2, CLIP).
- **Clustering e métricas**: documentação scikit-learn (k-means, silhouette, ARI/NMI).
- **Visualização**: documentação UMAP.
