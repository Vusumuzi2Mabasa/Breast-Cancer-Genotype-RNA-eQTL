args<-commandArgs(trailingOnly=TRUE);root<-if(length(args))args[1] else '.'
library(jsonlite)
files<-list.files(file.path(root,'investigations/eqtl_error/results'),pattern='pilot_fixtures[.]json$',full.names=TRUE)
matrixfrom<-function(a)do.call(rbind,lapply(a,unlist));errors<-numeric()
for(file in files)for(d in fromJSON(file,simplifyVector=FALSE)){
 x<-matrixfrom(d$x);x0<-x[,seq_len(d$p0),drop=FALSE];y<-unlist(d$y);w<-matrixfrom(d$multipliers)
 fit<-lm.fit(x0,y);h<-rowSums(qr.Q(qr(x0))^2);e<-fit$residuals/sqrt(1-h)
 for(k in 1:ncol(w)){
   ys<-fit$fitted.values+e*w[,k];a<-lm.fit(x0,ys);b<-lm.fit(x,ys)
   stat<-((sum(a$residuals^2)-sum(b$residuals^2))/(b$rank-a$rank))/(sum(b$residuals^2)/(length(y)-b$rank))
   errors<-c(errors,abs(stat-d$star_F[[k]]))
 }
}
stopifnot(max(errors)<1e-8)
write_json(list(status='passed',hypotheses_checked=length(errors),max_bootstrap_F_error=max(errors),R=as.character(getRversion())),file.path(root,'investigations/eqtl_error/results/R_validation.json'),auto_unbox=TRUE,pretty=TRUE,digits=16)
cat('Independent R wild bootstrap check passed\n')
