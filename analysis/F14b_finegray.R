# Fine-Gray subdistribution hazard models with death as a competing event (survival::finegray + weighted coxph).
# Input: cache/finegray_input.csv written by F14_tier3.py (M1 + gallstone-related covariates z*, signatures, per-outcome
# time T_*, status S_* (0 censored, 1 event, 2 death without the event) and eligibility elig_*).
suppressPackageStartupMessages({library(survival); library(data.table)})
P <- paste0(Sys.getenv("DIET_PROJECT_DIR", "."), "/")   # project directory (see README)
d <- fread(paste0(P, "cache/finegray_input.csv"))
z <- grep("^z", names(d), value = TRUE)
sigs <- c("sig_UPF", "sig_uPDI", "sig_hPDI", "sig_fibre")
outs <- c("Gallstones", "IBD", "Crohn", "UC", "Acute_pancreatitis", "AP_nonbiliary")
res <- list()
for (oc in outs) {
  dd <- d[get(paste0("elig_", oc)) == 1]
  dd[, status := factor(get(paste0("S_", oc)), levels = 0:2, labels = c("censor", "event", "death"))]
  dd[, time := get(paste0("T_", oc))]
  for (s in sigs) {
    f <- as.formula(paste("Surv(time, status) ~", paste(c(s, z), collapse = " + ")))
    fg <- finegray(f, data = as.data.frame(dd), etype = "event")
    m <- coxph(as.formula(paste("Surv(fgstart, fgstop, fgstatus) ~", paste(c(s, z), collapse = " + "))),
               data = fg, weights = fgwt)
    co <- summary(m)$coefficients[s, ]; ci <- summary(m)$conf.int[s, ]
    res[[length(res) + 1]] <- data.table(outcome = oc, x = s, n = nrow(dd), events = sum(dd$status == "event"),
                                         deaths = sum(dd$status == "death"), shr = ci["exp(coef)"],
                                         lci = ci["lower .95"], uci = ci["upper .95"], p = co["Pr(>|z|)"])
    cat(oc, s, sprintf("%.2f (%.2f-%.2f)", ci["exp(coef)"], ci["lower .95"], ci["upper .95"]), "\n")
  }
}
fwrite(rbindlist(res), paste0(P, "results/tier3_finegray.csv"))
