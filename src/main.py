from src.controllers.manager import Manager
from src.controllers.strategies.q_nodes import QNodes
from src.controllers.strategies.phi import Phi

def iniciar():
    """Punto de entrada principal"""
                    # ABCDEFGHIJ #
    estado_inicial = "1000000000"
    condiciones =    "1111111111"
    alcance =        "0111111001"
    mecanismo =      "0111111111"

    gestor_sistema = Manager(estado_inicial)

    ### Ejemplo de solución mediante módulo Q-Nodes ###
    analizador_fb = QNodes(gestor_sistema)
    sia_uno = analizador_fb.aplicar_estrategia(
         condiciones,
         alcance,
         mecanismo,
    )
    print(sia_uno)

    ### Ejemplo de solución mediante módulo Phi ###
    #analizador_fi = Phi(gestor_sistema)
    #sia_dos = analizador_fi.aplicar_estrategia(
    #    condiciones,
    #    alcance,
    #    mecanismo,
    #)
    #print(sia_dos)