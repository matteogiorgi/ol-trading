# Online Learning per l'allocazione dinamica tra strategie di trading

Questo progetto implementa un piccolo framework di backtest per combinare piu strategie semplici tramite un algoritmo di online learning di tipo Hedge, noto anche come Multiplicative Weights.

L'idea non e cercare ex ante la singola strategia migliore, ma aggiornare nel tempo i pesi assegnati a un insieme di esperti elementari in base ai risultati piu recenti. In un contesto non stazionario, questo approccio ha senso perche evita di fissare una sola regola su tutto il campione e consente un adattamento progressivo.

## 1. Obiettivo del progetto

Il codice costruisce una pipeline che:

- scarica una serie storica giornaliera di prezzi
- calcola i rendimenti semplici
- genera segnali da un insieme di esperti long-only in `{0, 1}`
- combina gli esperti con un learner Hedge
- stima il rendimento lordo e netto del portafoglio
- confronta il portafoglio con benchmark statici
- produce un report testuale e un report grafico persistenti

In pratica, il progetto serve a rispondere a questa domanda:

> una combinazione adattiva di strategie semplici riesce a produrre risultati competitivi rispetto a benchmark banali ma robusti?

## 2. Modello concettuale

### Dati

L'asset usato di default e `SPY`, con dati giornalieri scaricati tramite `yfinance`.

In [`data_loader.py`](/home/mattegiorgi/docs/ol-trading/data_loader.py) il download viene fatto con:

- `auto_adjust=True`, quindi il prezzo usato e gia adjusted
- selezione della colonna `Close`
- rinomina della colonna in `price`
- calcolo del rendimento semplice giornaliero

La formula del rendimento e:

$$
r_t = \frac{P_t - P_{t-1}}{P_{t-1}}
$$

### Esperti

Gli esperti sono definiti in [`experts.py`](/home/mattegiorgi/docs/ol-trading/experts.py). Ogni esperto restituisce un segnale binario:

$$
s_{t,i} \in \{0,1\}
$$

Interpretazione:

- `1`: l'esperto e investito sul mercato quel giorno
- `0`: l'esperto resta in cash

I rendimenti degli esperti sono quindi:

$$
r^{(expert)}_{t,i} = s_{t,i} \cdot r_t
$$

Questo significa che, nello stato attuale del progetto, non esistono posizioni short e non esiste leva.

Gli esperti di default sono:

- `buy_and_hold`: sempre investito
- `cash`: sempre fuori mercato
- `momentum_5`: investe se la somma dei rendimenti recenti a 5 giorni e positiva
- `momentum_20`: investe se la somma dei rendimenti recenti a 20 giorni e positiva
- `mean_reversion_3`: investe se la somma dei rendimenti recenti a 3 giorni e negativa
- `ma_crossover_5_20`: investe se la media mobile a 5 giorni e sopra quella a 20 giorni

### Learner Hedge

Il learner e implementato in [`learner.py`](/home/mattegiorgi/docs/ol-trading/learner.py).

Dato un vettore di pesi

$$
w_t = (w_{t,1}, \dots, w_{t,N}), \quad \sum_i w_{t,i} = 1
$$

il modello parte da pesi uniformi:

$$
w_{1,i} = \frac{1}{N}
$$

e aggiorna i pesi con:

$$
w_{t+1,i} = \frac{w_{t,i} e^{\eta r^{(expert)}_{t,i}}}{\sum_j w_{t,j} e^{\eta r^{(expert)}_{t,j}}}
$$

dove `eta` controlla quanto rapidamente il learner reagisce alle differenze di performance tra esperti.

Commento operativo:

- `eta` alto implica adattamento piu rapido e maggiore concentrazione dei pesi
- `eta` basso implica aggiornamenti piu lenti e una combinazione piu stabile

## 3. Logica del backtest

La logica centrale e in [`backtest.py`](/home/mattegiorgi/docs/ol-trading/backtest.py).

Per ogni data `t`, il codice:

1. usa solo lo storico fino a `t-1` per generare i segnali
2. osserva il rendimento realizzato nel giorno `t`
3. calcola i rendimenti di tutti gli esperti
4. applica i pesi correnti del learner
5. misura il turnover rispetto al giorno precedente
6. sottrae i costi di transazione dal rendimento lordo
7. aggiorna i pesi per il giorno successivo

Questa sequenza evita look-ahead bias nella generazione dei segnali: gli esperti vedono soltanto dati passati.

### Rendimento del portafoglio

Il rendimento giornaliero lordo del portafoglio e:

$$
r_t^{(portfolio)} = \sum_{i=1}^{N} w_{t,i} r^{(expert)}_{t,i}
$$

### Turnover e costi di transazione

Il turnover giornaliero e:

$$
turnover_t = \sum_i |w_{t,i} - w_{t-1,i}|
$$

e il rendimento netto diventa:

$$
r_t^{(net)} = r_t^{(portfolio)} - c \cdot turnover_t
$$

Nel file [`main.py`](/home/mattegiorgi/docs/ol-trading/main.py) il costo di transazione di default e `0.0005`.

## 4. Benchmark implementati

Il progetto confronta la strategia Hedge con tre benchmark:

- `Buy and Hold`: il rendimento dell'esperto sempre investito
- `Equal Weight Experts`: media semplice dei rendimenti degli esperti
- `Best Expert in Hindsight`: l'esperto con la miglior performance cumulata ex post

Il benchmark "best expert in hindsight" non e investibile in tempo reale, ma e utile come upper bound descrittivo sul set di esperti disponibile.

## 5. Metriche calcolate

Le metriche sono implementate in [`metrics.py`](/home/mattegiorgi/docs/ol-trading/metrics.py).

Per ciascuna serie di rendimenti vengono calcolate:

- `cumulative_return`
- `sharpe_ratio`
- `annualized_volatility`
- `max_drawdown`
- `average_turnover` quando viene fornita la serie di turnover

### Ricchezza cumulata

La curva di wealth e costruita come:

$$
V_t = V_0 \prod_{\tau=1}^{t}(1 + r_\tau)
$$

con `V_0 = 1`.

### Sharpe ratio

La versione implementata e annualizzata, con tasso risk-free pari a zero per default:

$$
Sharpe = \frac{\mathbb{E}[r_t]}{\sigma(r_t)} \cdot \sqrt{252}
$$

### Maximum drawdown

Il drawdown e misurato come distanza percentuale dal massimo storico della curva di wealth.

Commento importante:

- il drawdown viene calcolato, ma al momento non viene plottato come figura separata
- nel README precedente era citato anche un grafico di drawdown e una sensitivity su `eta`, ma il codice corrente non produce ancora quei due output

## 6. Esecuzione del progetto

### Requisiti

Le dipendenze principali sono:

- `pandas`
- `numpy`
- `matplotlib`
- `yfinance`

Sono elencate in [`requirements.txt`](/home/mattegiorgi/docs/ol-trading/requirements.txt).

### Avvio

Per eseguire il progetto:

```bash
python main.py
```

I parametri principali sono attualmente definiti direttamente in [`main.py`](/home/mattegiorgi/docs/ol-trading/main.py#L33):

- `ticker = "SPY"`
- `start = "2015-01-01"`
- `end = "2025-01-01"`
- `eta = 5.0`
- `transaction_cost = 0.0005`

Questa scelta rende il progetto semplice da leggere, ma significa anche che al momento non esiste ancora una CLI parametrica con argomenti da linea di comando.

## 7. Output generati

Una delle modifiche recenti e che l'esecuzione non si limita piu a mostrare output a schermo: salva anche artefatti persistenti in `outputs/`.

### Report testuale

Il programma stampa a terminale una tabella di metriche per:

- Hedge Portfolio (gross)
- Hedge Portfolio (net)
- Buy and Hold
- Equal Weight Experts
- Best Expert in Hindsight

Lo stesso contenuto viene salvato in un file `.txt`, per esempio:

[`outputs/spy_2015-01-01_2025-01-01_metrics.txt`](/home/mattegiorgi/docs/ol-trading/outputs/spy_2015-01-01_2025-01-01_metrics.txt)

Questo e utile per:

- conservare i risultati di una run
- confrontare run diverse
- includere facilmente l'output in una relazione o in appendice

### Report grafico

I plot non vengono piu mostrati solo con `plt.show()`, ma salvati in un PDF multipagina-friendly contenente la figura composta. Il file ha forma:

[`outputs/spy_2015-01-01_2025-01-01_plots.pdf`](/home/mattegiorgi/docs/ol-trading/outputs/spy_2015-01-01_2025-01-01_plots.pdf)

Nello stato attuale la figura contiene tre pannelli:

- curva di wealth cumulata del portafoglio e dei benchmark
- evoluzione nel tempo dei pesi degli esperti
- turnover giornaliero

Commento pratico:

- questo approccio e piu adatto a esecuzioni batch o da terminale
- evita di dipendere dall'apertura interattiva di finestre grafiche
- rende piu semplice archiviare i risultati di backtest multipli

## 8. Struttura del codice

Il progetto e suddiviso in moduli con responsabilita abbastanza chiare:

- [`data_loader.py`](/home/mattegiorgi/docs/ol-trading/data_loader.py): download prezzi e calcolo rendimenti
- [`experts.py`](/home/mattegiorgi/docs/ol-trading/experts.py): definizione degli esperti e factory degli esperti di default
- [`learner.py`](/home/mattegiorgi/docs/ol-trading/learner.py): algoritmo Hedge
- [`backtest.py`](/home/mattegiorgi/docs/ol-trading/backtest.py): simulazione storica e benchmark
- [`metrics.py`](/home/mattegiorgi/docs/ol-trading/metrics.py): metriche di performance
- [`main.py`](/home/mattegiorgi/docs/ol-trading/main.py): orchestrazione end-to-end, stampa report e salvataggio output

## 9. Osservazioni metodologiche

Il progetto e coerente come prototipo didattico o base per una tesi breve, ma e importante essere espliciti su alcuni limiti.

### Cosa fa bene

- separa bene dati, esperti, learner, backtest e metriche
- evita look-ahead bias nella generazione dei segnali
- include turnover e costi di transazione
- confronta il learner con benchmark semplici ma utili
- salva in modo riproducibile sia output testuale sia grafico

### Limiti attuali

- un solo asset
- soli segnali binari `{0,1}`
- nessuna posizione short
- nessun costo fisso o slippage avanzato
- nessuna validazione out-of-sample o walk-forward
- parametri definiti a mano in `main.py`
- nessuna analisi di sensitivita sistematica su `eta`

Questi limiti non invalidano il progetto, ma ne definiscono il perimetro: e un laboratorio compatto per testare online learning su esperti semplici, non ancora un motore di ricerca quantitativa completo.

## 10. Possibili estensioni

Gli sviluppi piu naturali sono:

- aggiungere segnali in `{-1, 0, 1}` per introdurre strategie short/flat/long
- supportare piu asset e allocazione cross-sectional
- esporre i parametri via CLI
- salvare anche i risultati numerici in CSV o parquet
- aggiungere drawdown plot e confronto su piu valori di `eta`
- introdurre forgetting factor o altri algoritmi online
- separare training, validation e test in modo piu rigoroso

## 11. Messaggio centrale

Il punto piu interessante del progetto e questo:

> invece di trattare le strategie come alternative mutuamente esclusive, le si puo trattare come esperti concorrenti e aggiornare dinamicamente la fiducia assegnata a ciascuna.

Questa e una formulazione semplice, interpretabile e facilmente estendibile dell'idea di adattamento online nei mercati finanziari.
