import sys
import os
from vs.environment import Env
from vs.constants import VS
from explorer import Explorer
from rescuer import Rescuer

def main(data_folder_name, config_ag_folder_name):
    # Obter caminhos para pastas de dados e configurações
    current_folder = os.path.abspath(os.path.dirname(__file__))
    config_ag_folder = os.path.abspath(os.path.join(current_folder, config_ag_folder_name))
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))
    
    # Criar uma instância do ambiente
    env = Env(data_folder)
    
    # Instanciar o agente mestre socorrista
    rescuer_file = os.path.join(config_ag_folder, "rescuer_1_config.txt")
    master_rescuer = Rescuer(env, rescuer_file, 4)

    # Instanciar os exploradores
    for exp in range(1, 5):
        filename = f"explorer_{exp:1d}_config.txt"
        explorer_file = os.path.join(config_ag_folder, filename)
        Explorer(env, explorer_file, master_rescuer)

    env.run()

if __name__ == '__main__':
    # Verificar se há argumentos de linha de comando para as pastas de dados e configurações
    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        data_folder_name = os.path.join("..", "datasets", "data_300v_90x90")
        config_ag_folder_name = os.path.join("..","config")
        
    main(data_folder_name, config_ag_folder_name)