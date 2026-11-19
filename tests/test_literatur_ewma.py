"""Literatur-Referenzwerte (nur als eigene Nachrechnung): Paare (λ, L) mit ARL_0 ≈ 500 aus Lucas/Saccucci (1990), Technometrics 32(1), Tabelle 3.

Lesestufe: im Quellenbericht der Linienplanung nur als Text mit Zeichenverlusten (Scan) geführt, die Vormessung nennt die Quelle „nicht lesbar“. Darum steht hier KEINE Tabellenzahl der Quelle
als Beleg, sondern nur: mit diesen (λ, L) rechnet die Demo ARL_0 = 500 auf ±0,5. Tabellenzahlen von NIST und die Verzögerungsspalten von Moustakides u. a. (2009) sind fehlerhaft und stehen nicht im Test."""

import pytest

import ewk_chains as K


@pytest.mark.parametrize("lam,L,arl", [(0.25, 2.998, 499.836), (0.10, 2.814, 499.580), (0.05, 2.615, 499.933)])
def test_paare_mit_arl0_500_als_eigene_nachrechnung(lam, L, arl):
    a = K.arl_ewma(lam, L * K.sigma_z(lam), 0.0, True)
    assert abs(a - 500) < 0.5  # 500 auf ±0,5
    assert a == pytest.approx(arl, abs=2e-3)
