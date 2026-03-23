# Online Learning per l’allocazione dinamica tra strategie di trading

## 1. Obiettivo del progetto

L’obiettivo è costruire un sistema che:

- osserva i rendimenti prodotti da più strategie di trading semplici (esperti)
- aggiorna dinamicamente il peso assegnato a ciascuna strategia
- costruisce un portafoglio finale come combinazione pesata delle strategie
- confronta questa combinazione con benchmark statici

Idea centrale:

> Invece di cercare una singola strategia ottimale, si costruisce un algoritmo che adatta dinamicamente la fiducia tra più strategie.

---

## 2. Domanda di ricerca

Un algoritmo di online learning è in grado di combinare dinamicamente strategie di trading semplici e ottenere performance competitive rispetto a benchmark statici?

---

## 3. Ipotesi

In un contesto finanziario non stazionario, una combinazione adattiva di strategie semplici può risultare più robusta rispetto a una strategia statica.

---

## 4. Dataset

- Asset: SPY (consigliato)
- Frequenza: giornaliera
- Periodo: es. 2015–2025
- Dati: Adjusted Close

### Returns

$$
r_t = \frac{P_t - P_{t-1}}{P_{t-1}}
$$

---

## 5. Esperti (strategie base)

### 5.1 Buy and Hold
$$
s_{t,1} = 1
$$

### 5.2 Cash
$$
s_{t,2} = 0
$$

### 5.3 Momentum (5 giorni)
$$
s_{t,3} =
\begin{cases}
1 & \text{se } \sum_{k=1}^{5} r_{t-k} > 0 \\
0 & \text{altrimenti}
\end{cases}
$$

### 5.4 Momentum (20 giorni)
$$
s_{t,4} =
\begin{cases}
1 & \text{se rendimento a 20 giorni positivo} \\
0 & \text{altrimenti}
\end{cases}
$$

### 5.5 Mean Reversion (3 giorni)
$$
s_{t,5} =
\begin{cases}
1 & \text{se } \sum_{k=1}^{3} r_{t-k} < 0 \\
0 & \text{altrimenti}
\end{cases}
$$

### 5.6 Moving Average Crossover
$$
s_{t,6} =
\begin{cases}
1 & \text{se } MA_5(t-1) > MA_{20}(t-1) \\
0 & \text{altrimenti}
\end{cases}
$$

---

## 6. Rendimenti degli esperti

Ogni esperto produce un segnale:

$$
s_{t,i} \in \{0,1\}
$$

Rendimento:

$$
r_{t,i}^{(expert)} = s_{t,i} \cdot r_t
$$

---

## 7. Algoritmo di Online Learning (Hedge)

Pesi:

$$
w_t = (w_{t,1}, \dots, w_{t,N}), \quad \sum_i w_{t,i} = 1
$$

Inizializzazione:

$$
w_{1,i} = \frac{1}{N}
$$

Aggiornamento:

$$
w_{t+1,i} = \frac{w_{t,i} e^{\eta r_{t,i}^{(expert)}}}{\sum_j w_{t,j} e^{\eta r_{t,j}^{(expert)}}}
$$

---

## 8. Rendimento del portafoglio

$$
r_t^{(portfolio)} = \sum_{i=1}^N w_{t,i} r_{t,i}^{(expert)}
$$

---

## 9. Costi di transazione

Turnover:

$$
turnover_t = \sum_i |w_{t,i} - w_{t-1,i}|
$$

Rendimento netto:

$$
r_t^{(net)} = r_t^{(portfolio)} - c \cdot turnover_t
$$

---

## 10. Benchmark

- Buy and Hold
- Equal-weight tra esperti
- Best expert in hindsight

---

## 11. Metriche

### 11.1 Cumulative Return

$$
V_t = V_{t-1}(1 + r_t)
$$

### 11.2 Sharpe Ratio

$$
Sharpe = \frac{\text{mean}(r_t)}{\text{std}(r_t)} \cdot \sqrt{252}
$$

### 11.3 Maximum Drawdown

Massima perdita rispetto al massimo precedente.

### 11.4 Volatilità

$$
\sigma_{ann} = \text{std}(r_t)\sqrt{252}
$$

### 11.5 Turnover medio

### 11.6 Hit Ratio (opzionale)

---

## 12. Backtest (pipeline)

Per ogni giorno t:

1. Genera segnali usando dati fino a t-1
2. Calcola rendimenti degli esperti
3. Calcola rendimento del portafoglio
4. Applica costi
5. Aggiorna pesi
6. Salva risultati

---

## 13. Attenzione: No Look-Ahead Bias

- I segnali devono usare solo dati passati
- Il rendimento è quello del giorno successivo

---

## 14. Output grafici

- Cumulative wealth
- Pesi nel tempo
- Drawdown
- Sensitivity su η

---

## 15. Risultati attesi

- comportamento adattivo
- possibile riduzione del drawdown
- performance comparabile ai benchmark

---

## 16. Estensioni

- Short/long ($\{-1,0,1\}$)
- Più asset
- Forgetting factor
- Confronto con altri algoritmi online

---

## 17. Struttura del codice

- data_loader.py
- experts.py
- learner.py
- backtest.py
- metrics.py
- main.py

---

## 18. Messaggio centrale

> Un algoritmo di online learning può adattare dinamicamente la combinazione di strategie di trading in presenza di mercati non stazionari.

---

## 19. Limiti

- Strategie base semplici
- Backtest dipendente dai dati
- Sensibilità a parametri (η, costi)
- Non garantisce performance future

---

## 20. Piano di lavoro

1. Dataset e returns
2. Esperti base
3. Hedge learner
4. Backtest
5. Metriche
6. Costi e analisi finale
