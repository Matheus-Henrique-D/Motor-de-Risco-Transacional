import kagglehub

print("Iniciando download do dataset 'home-credit-default-risk' do Kaggle...")
try:
    path = kagglehub.competition_download('home-credit-default-risk')
    print("\n[OK] Download concluído com sucesso!")
    print("Arquivos baixados em:", path)
    print("\nPara usar no projeto:")
    print("Copie todos os arquivos dessa pasta acima para a pasta 'home-credit-default-risk' na raiz do seu repositório.")
except Exception as e:
    print("\n[ERRO] Erro durante o download:")
    print(str(e))
    print("\nInstruções de Resolução:")
    print("1. Certifique-se de ter uma conta no Kaggle (https://www.kaggle.com).")
    print("2. Vá na página da competição (https://www.kaggle.com/c/home-credit-default-risk) e aceite as regras clicando em 'Rules' ou 'Join Competition'.")
    print("3. Crie um token de API em seu perfil do Kaggle (Configurações -> seção 'API' -> clique em 'Create New Token'). Isso baixará um arquivo chamado 'kaggle.json'.")
    print("4. Mova esse arquivo 'kaggle.json' para a pasta 'C:\\Users\\natha\\.kaggle\\kaggle.json' e tente rodar este script novamente.")
