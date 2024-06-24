#preparer météo data en se basant sur le script record_raw_weather.py
library(tcltk2)
library(lubridate)
library(REdaS)
library(reshape2)
library(tidyverse)
library(dplyr)
library(plyr)

fichier<-file.choose()
ex_meteo<- read.table(fichier,sep=c(","),dec=".",header = TRUE
                      )


ex_meteo = read.table(fichier,sep=c(","),dec=".",header = TRUE
)%>%mutate(DATE = as.Date(Time))

# date_tr<-colsplit(string=as.character(ex_meteo$date), pattern="T", names=c("Part1", "Part2"))
# ex_meteo<-cbind(date_tr[,1],ex_meteo)

# colnames(ex_meteo)[1] <- "DATE"
# colnames(ex_meteo)[7] <- "RELATIVE_HUMIDITY"


#Temperature moyenne

# meteo_data<-ddply(ex_meteo, .(DATE), summarize,  RAIN_FALL=round(mean(as.numeric(RAIN_FALL),digits = 3)), TEMPERATURE_MAX=round(max(TEMPERATURE_MAX),digits = 3),TEMPERATURE_MIN=round(mean(TEMPERATURE_MIN),digits = 3),TEMPERATURE_AVG=round(mean(TEMPERATURE),digits = 3),RELATIVE_HUMIDITY=round(mean(HUMIDITY),digits = 3),WIND_SPEED=round(mean(WIND_SPEED),digits = 3))
   
    meteo_data<-ddply(ex_meteo, .(DATE), summarize,  RAIN_FALL=round(max(as.numeric(RAIN_FALL)),digits = 3),  TEMPERATURE_MAX=round(max(TEMPERATURE),digits = 3),TEMPERATURE_MIN=round(min(TEMPERATURE),digits = 3),TEMPERATURE_AVG=round(mean(TEMPERATURE),digits = 3),RELATIVE_HUMIDITY=round(mean(HUMIDITY),digits = 3),WIND_SPEED=round(mean(WIND_SPEED),digits = 3))
    
    meteo_result<-data.frame(meteo_data)%>%drop_na()
    
    colnames(meteo_result)<-c("DATE","RAIN_FALL", "TEMPERATURE_MAX","TEMPERATURE_MIN","TEMPERATURE_AVG","RELATIVE_HUMIDITY","WIND_SPEED")
    #directory<-setwd("/home/lamiaa/Bureau/")
    
    write.table(meteo_result, file="2021_Arvalis_Maïs_31-10.csv",append = FALSE, quote = TRUE, sep = ",",
              eol = "\n", na = "NA", dec = ".", row.names = FALSE,
              col.names = TRUE, qmethod = c("escape", "double"),
              fileEncoding = "")

### référence utilisée https://edis.ifas.ufl.edu/pdf/AE/AE45900.pdf
#météo data
library(tcltk2)
library(lubridate)
library(REdaS)
fichier<-file.choose()
ex_meteo_sanspluie <- read.csv2(fichier,sep=";",dec=",",header = TRUE)
# calcul Mean saturation vapour pressure
## e°(T) saturation vapour pressure at the air temperature T [kPa]
## http://www.fao.org/3/x0490e/x0490e07.htm#measurement
e0<-rep(0,length(ex_meteo_sanspluie$DATE))
for(i in 1:(length(ex_meteo_sanspluie$DATE)))
{
  e0[i]=0.6108*exp((17.27*ex_meteo_sanspluie$TEMPERATURE_mean[i])/(ex_meteo_sanspluie$TEMPERATURE_mean[i]+237.3))
}
# calcul Slope of saturation vapour pressure curve (D )
delta<-rep(0,length(ex_meteo_sanspluie$DATE))
for(i in 1:(length(ex_meteo_sanspluie$DATE)))
{
  delta[i]<-(4098*e0[i])/(ex_meteo_sanspluie$TEMPERATURE_mean[i]+237.3)^2
}
# Atmospheric Pressure (P)
#z = elevation above sea level, m
z=50 # marguerittes)
  P<-101.3*((293-0.065*z)/293)^5.26
#Psychrometric constant
psy<-0.00065*P
#Delta Term (DT) (auxiliary calculation for Radiation Term)
#The delta term is used to calculate the “Radiation Term” of the overall ET equation (Eq. 33)
DT<-rep(0,length(ex_meteo_sanspluie$DATE)) 
for(i in 1:length(ex_meteo_sanspluie$DATE))
{
  
  DT[i] = delta[i]/(delta[i]+psy*(1+0.34*ex_meteo_sanspluie$WIND_SPEED[i]))
}
###  Psi Term (PT) (auxiliary calculation for Wind Term) The psi term is used to calculate the “Wind Term” of the overall ETo equation [Eq. 34]
PT<-rep(0,length(ex_meteo_sanspluie$DATE)) 
for(i in 1:length(ex_meteo_sanspluie$DATE))
{
  
  PT[i] = psy/(delta[i]+psy*(1+0.34*ex_meteo_sanspluie$WIND_SPEED[i]))
}
#  Temperature Term (TT) (auxiliary calculation for Wind Term)
TT<-rep(0,length(ex_meteo_sanspluie$DATE)) 
for(i in 1:length(ex_meteo_sanspluie$DATE))
{
  Temp_mean<-(ex_meteo_sanspluie$TEMPERATURE_MAX[i] - ex_meteo_sanspluie$TEMPERATURE_MIN[i])/2
  TT[i] = (900/(Temp_mean+273))*ex_meteo_sanspluie$WIND_SPEED[i]
}


# Determine daily extraterrestrial radiation. Ra
dr<-rep(0,length(ex_meteo_sanspluie$DATE)) # inverse relative distance Earth-Sun
g_sc = 0.082  # Solar constant (MJ /m² /min)
for(i in 1:length(ex_meteo_sanspluie$DATE))
{
  J<-yday(as.Date(ex_meteo_sanspluie$DATE[i]))
  # J is the number of the day in the year between 1 (1 January) and 365 or 366 (31 December).
  dr[i] = 1 + 0.033 * cos(2 * pi / 365 * J)  # Inverse distance Earth-Sun (23)
}
#lat rodilhan 43.827
lat=43.827
phi = deg2rad(lat)  # Latitude
# d solar decimation Equation 24 
dd<-rep(0,length(ex_meteo_sanspluie$DATE))
for(i in 1:length(ex_meteo_sanspluie$DATE))
{
  J<-yday(as.Date(ex_meteo_sanspluie$DATE[i]))
    dd[i]<-0.409*sin((((2*pi)/365)*J)-1.39)
}
# ws sunset hour angle w s = arccos [-tan (j) tan (d)] (25)
Ws<-rep(0,length(ex_meteo_sanspluie$DATE))
for(i in 1:length(ex_meteo_sanspluie$DATE))
{
  Ws[i]<-acos(-tan(phi)*tan(dd[i]))
}

#Extraterrestrial radiation for daily periods (Ra) eq 21 (FAO) 

ra<-rep(0,length(ex_meteo_sanspluie$DATE))
for(i in 1:length(ex_meteo_sanspluie$DATE))
{
  ra[i]<-((24*60)/pi)*g_sc*dr[i]*(Ws[i]*sin(phi)*sin(dd[i])+cos(phi)*cos(dd[i])*sin(Ws[i]))
  if (ra[i]<0)
    ra[i]=0
}
# calcul ET0 _méthode 1 eq 52 (http://www.fao.org/3/x0490e/x0490e07.htm#an%20alternative%20equation%20for%20eto%20when%20weather%20data%20are%20missing)
ET0_1<-rep(0,length(ex_meteo_sanspluie$DATE))
for(i in 1:length(ex_meteo_sanspluie$DATE))
{
  Temp_mean<-(ex_meteo_sanspluie$TEMPERATURE_MAX[i] - ex_meteo_sanspluie$TEMPERATURE_MIN[i])/2
  
  ET0_1[i]<-0.0023*(ex_meteo_sanspluie$TEMPERATURE_mean[i]+17.8)*Temp_mean *ra[i]
}

# Determine the net radiation.
rs<-rep(0,length(ex_meteo_sanspluie$DATE))
rns<-rep(0,length(ex_meteo_sanspluie$DATE))
rnl<-rep(0,length(ex_meteo_sanspluie$DATE))
rn<-rep(0,length(ex_meteo_sanspluie$DATE))
for(i in 1:length(ex_meteo_sanspluie$DATE))
{
  if(ra[i]==0)
    rs[i]=0
  else
  {
    #rs solar radiation  eq 50
    rs[i]=0.19*sqrt(ex_meteo_sanspluie$TEMPERATURE_MAX[i]-ex_meteo_sanspluie$TEMPERATURE_MIN[i])*ra[i]
  }
  #Compute net shortwave radiation (rns)
  rns[i]<-(1-0.23)*rs[i]
  ## Compute net longwave radiation (rnl) (Eq. 39)
  rnl[i]<-(0.34-0.14*sqrt(e0[i]))*(1.35*rs[i]/(0.75*ra[i])-0.35)
  #  rn is the difference between the incoming rns and the outgoing rnl
  rn[i]<-rns[i]-rnl[i]
}
### Rng To express the net radiation (Rn) in equivalent of evaporation (mm) (Rng);
rng<-rep(0,length(ex_meteo_sanspluie$DATE))

for(i in 1:length(ex_meteo_sanspluie$DATE))
{
  rng[i]<-0.408*rn[i]
}


### calcul ETP 
gam = 0.067  # Link to altitude (Table 2.2)
e_max<-rep(0,length(ex_meteo_sanspluie$DATE))
e_min<-rep(0,length(ex_meteo_sanspluie$DATE))
es<-rep(0,length(ex_meteo_sanspluie$DATE))
ea<-rep(0,length(ex_meteo_sanspluie$DATE))
ETP<-rep(0,length(ex_meteo_sanspluie$DATE))

for(i in 1:(length(ex_meteo_sanspluie$DATE)))
{
  e_min[i]=0.6108*exp((17.27*ex_meteo_sanspluie$TEMPERATURE_MIN[i])/(ex_meteo_sanspluie$TEMPERATURE_MIN[i]+237.3))
  e_max[i]=0.6108*exp((17.27*ex_meteo_sanspluie$TEMPERATURE_MAX[i])/(ex_meteo_sanspluie$TEMPERATURE_MAX[i]+237.3))
  es[i]<-( e_min[i]+e_max[i])/2
  ea[i]<-(es[i]*ex_meteo_sanspluie$RELATIVE_HUMIDITY[i])/100
  ETP[i]<-((0.408*delta[i]*rn[i])+gam*(900/(ex_meteo_sanspluie$TEMPERATURE_mean[i]+273)*ex_meteo_sanspluie$WIND_SPEED[i]*(es[i]-ea[i])))/(delta[i]+gam*(1+0.34*ex_meteo_sanspluie$WIND_SPEED[i]))
}


# ######## Radiation term (ETrad)
# 
# ETrad<-rep(0,length(ex_meteo_sanspluie$DATE))
# 
# for(i in 1:length(ex_meteo_sanspluie$DATE))
# {
#   ETrad[i]<-delta[i]*rng[i]
# }
# ###Wind term (ETwind)
# 
# ETwind<-rep(0,length(ex_meteo_sanspluie$DATE))
# 
# for(i in 1:length(ex_meteo_sanspluie$DATE))
# {
#   ETwind[i]<-PT[i]*TT[i]*(es[i]-ea[i])
# }
# 
# for(i in 1:(length(ex_meteo_sanspluie$DATE)))
# {
#   
#   ET0[i]<-ETrad[i]+ETwind[i]
# }




    date<-as.Date((ex_meteo_sanspluie$DATE),format="%d/%m/%Y")
    
    Resultat4<-data.frame(date,ex_meteo_sanspluie$TEMPERATURE_mean, ex_meteo_sanspluie$RAIN_FALL,round(ETP,digits = 3))
    colnames(Resultat4)<-c("Date","Temp","Pluie","ETP")
    
    write.table(Resultat4, file = "prevision_Rodilhan_2021_etp.csv", append = FALSE, quote = TRUE, sep = ";",
                eol = "\n", na = "NA", dec = ",", row.names = FALSE,
                col.names = TRUE, qmethod = c("escape", "double"),
                fileEncoding = "")

######################### jointure de deux dataframe
# Meteo historique
library(tcltk2)
fichier<-file.choose()
ex_meteo_sanspluie <- read.csv2(fichier,sep=";",dec=".",header = TRUE)

# Meteo prevision
library(tcltk2)
fichier1<-file.choose()
ex_meteo_prevision <- read.csv2(fichier1,sep=";",dec=".",header = TRUE)


###jointure de deux dataframe
All_meteo<-rbind(ex_meteo_sanspluie,ex_meteo_prevision)

#colnames(All_meteo)<-c("Date","Temp","Pluie")

write.table(All_meteo, file = "All_Data_météo_Rodilhan_2021_merged.csv", append = FALSE, quote = TRUE, sep = ";",
            eol = "\n", na = "NA", dec = ".", row.names = FALSE,
            col.names = TRUE, qmethod = c("escape", "double"),
            fileEncoding = "")
