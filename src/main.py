from src.controllers.manager import Manager

from src.controllers.strategies.force import BruteForce
from src.controllers.strategies.q_nodes import QNodes


def iniciar():
    """Punto de entrada principal"""
                    # ABCDEFGHIJ #
    estado_inicial = "1000000000"
    condiciones =    "1111111111"
                    #ABCDEFGHIJ_{t+1} si alguna letra no aparece es 0
    alcance =        "1111111111"
                    #ABCDEFGHIJ_{t}
    mecanismo =      "1111111111"

    gestor_sistema = Manager(estado_inicial)

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    analizador_fb = QNodes(gestor_sistema)
    sia_uno = analizador_fb.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
    )
    print(sia_uno)

   


