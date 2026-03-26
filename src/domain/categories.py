"""
Catálogo de categorías de soporte TI y reglas de enrutamiento.

Este módulo es conocimiento puro del dominio: no depende de ningún
framework (FastAPI, PyTorch, sklearn). Puede ser importado por
cualquier capa sin introducir acoplamiento.
"""

from dataclasses import dataclass

CONFIDENCE_THRESHOLD: float = 0.40


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    team: str


CATEGORIES: list[Category] = [
    Category("red_conectividad", "Red / Conectividad", "Equipo Redes"),
    Category("acceso_contrasenas", "Acceso / Contraseñas", "Soporte N1 / IAM"),
    Category("correo_office365", "Correo / Office 365", "Equipo O365"),
    Category("impresion_perifericos", "Impresión / Periféricos", "Equipo Periféricos"),
    Category("aplicacion_errores", "Aplicación / Errores", "Equipo Aplicaciones"),
    Category("hardware_equipo", "Hardware / Equipo", "Equipo Hardware"),
    Category("vpn_remoto", "VPN / Remoto", "Equipo Seguridad/Redes"),
    Category("otros", "Otros / No clasifica", "Revisión humana"),
]

CATEGORY_KEYS: list[str] = [c.key for c in CATEGORIES]

_by_key: dict[str, Category] = {c.key: c for c in CATEGORIES}
_by_label: dict[str, Category] = {c.label: c for c in CATEGORIES}


def get_category(key: str) -> Category:
    """Retorna la categoría por su clave. Lanza KeyError si no existe."""
    return _by_key[key]


def get_team(key: str) -> str:
    """Retorna el equipo sugerido para una clave de categoría."""
    return _by_key[key].team


def get_category_by_label(label: str) -> Category:
    """Retorna la categoría a partir de su etiqueta legible."""
    return _by_label[label]
