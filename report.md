# Kursprojekt - Rapport

*Viktor Nylund - Beslutsstödsystem*

- summering, hur gick det
- resultat, screenshot/kodblock
- reflektion, jag lärde mig.., jag tycker..,

## Innehållsförteckning

- [Process](#process)
 - [Innehållsbaserad filtreringen](#innehållsbaserad-filtreringen)
 - [Kollaborativ filtrering](#kollaborativ-filtrering)
 - [Hybrid rekommendation](#hybrid-rekommendation)
- [Resultat](#resultat)
- [Evaluering](#evaluering)

## Process

### **Innehållsbaserad filtreringen**  
Använder sig av spelens olika attribut, så som platform och pris, för att hitta likheter mellan spelen. Tyvärr finns inte genre i mitt data. 

### **Kollaborativ filtrering**  
Använder sig av användarnas rekommendationer och de spel de spelat för att hitta likheter mellan användarna och spel de antagligen skulle tycka om.

Jag kombinerar dessa för att skapa ett hybridsystem som använder sig av båda faktorerna. Den innehållsbaserade listan sätts till med de kollaborativa rekommendationerna.
  
### **Hybrid rekommendation**

På grund av storleken på users filen (13781059 users) tog programet rätt så länge att köra.  
Detta var resultaten när jag begränsade det till att endast använda 5 users data. 

<img src="images/out1.png" alt="Screenshot of first output" width="700"/>

<img src="images/code1.png" alt="Screenshot of filtering code" width="700"/>

Jag bestämde att begränsa det till att använda users med minst 5 ratings för att se om det är mer hanterligt. 
Därtill berättade AI att programmet blir snabbare med `group()` och att använda en gläsare 'sparse' user-item matris med hjälp av `scipy.sparse`. Denna använder mindre minne och gör träningen snabbare. Jag lade även till `user_reviews, price_final och discount` till features i hopp om att de ger bättre resultat. 

<img src="images/imp1.png" alt="Screenshot of filtering code" width="700"/>

Jag bestämde också att helt enkelt minska datamängden direkt till ett par tusen. 
Men jag ville också hålla datan så autentisk som möjligt, och vill inte endast använda mig av de spel med flest rekommendationer. Även om det försämrar inlärning, ville jag att systemet inte skulle förlita sig helt på användare som rekommenderat många spel.  
Därför är 1/4 av den data jag använder mig av samplad på måfå, och den återstående andelen är användare med minst 5 rekommendationer.

<img src="images/code2.png" alt="Screenshot of data splitting code" width="700"/>

| Metric                  | Value      |
|-------------------------|------------|
| Användare               | 13,781,059 |
| >=5 rekommendationer    | 879,211    |
| Kombinerad data         | 952,526    |
| Unika användare         | 99,844     |

Det behövs evaluering för att veta säkert om detta är bättre eller inte. Det borde förbättra systemet ifall om användaren har väldigt få spel i sin lista. 

---

## Resultat

**Första Hybrid (halvt söndrig)**

<img src="images/out2.png" alt="Screenshot of second output" width="700"/>

Förändringarna ovan gjorde programet mycket snabbare att köra. Det krävdes bara 2-3 minuter.  
Vid första anblick blir jag dock något oroad att rekommendationssystemet försämrats, men många av spelen i listan är obekanta för mig. 
T.ex. finns inte `The Binding of Isaac` kvar i listan, som var en passande rekommendation. `INSIDE`, uppföljaren, lyser även med sin frånvaro.  
AI menade att `discount` kan vara en vilseledande feature, så jag exkluderade den. Jag ändrade också andelen data som samplades på måfå från 1/4 till 1/6. Detta så att andelen slumpad data inte har för stor inverkan. Jag ökade även gränsen för ratings till minst 10 per användare, vilket krävde att jag minskade mängden använd data en aning.  
Resultatet var nära på det samma.

<img src="images/out3.png" alt="Screenshot of third output" width="700"/>

I detta skede modulariserade jag koden helt så att jag kunde köra båda funktionerna separat lättare.

<img src="images/user1.png" alt="Screenshot of bad user" width="700"/>

Här ser vi att användaren har väldigt få rader med rekommendationer. Användare `4616950` med 36 recensioner är bättre lämpad. Jag såg även till att manuellt sätta med användaren i mitt testdata. 

| Metric                  | Value      |
|-------------------------|------------|
| Användare               | 13,781,059 |
| >=10 rekommendationer   | 1,276,399  |
| Kombinerad data         | 1,305,241  |
| Unika användare         | 69,961     |

**Innehållsbaserad rekommendation**

<img src="images/out4.png" alt="Screenshot of content based output" width="700"/>

Skillnaden beror endast på att det nu printar 10 rekommendationer, som det borde göra. Annars är 

**Kollaborativ rekommendation**

<img src="images/out5.png" alt="Screenshot of collaborative output" width="700"/>

Här ser vi helt andra typer av spel, som borde vara helt baserade på användarens tidigare recensioner. 

**Andra Hybrid (fixad)**

<img src="images/out6.png" alt="Screenshot of hybrid output" width="700"/>

Här ser vi precis samma spel som i den kollaborativa filtreringen. Detta p.g.a. att denna lista lägs till först och de innehållsbaserade rekommendationerna lägs till därefter. Detta är en simpel hybrid model.  
AI gav detta som enkelt förslag på en viktad rekommenderare som "blandar" listorna.

<img src="images/code3.png" alt="Screenshot of weighted hybrid code" width="700"/>

<img src="images/out7.png" alt="Screenshot of weighted hybrid output" width="700"/>

Här syns 7 nya spel, vilket är som förväntat från `alpha = 0.7`. 

## Evaluering




---


