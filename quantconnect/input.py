#cell 0
import os
from hurst import compute_Hc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.cluster import OPTICS
from sklearn.decomposition import PCA
from statsmodels.api import OLS
import statsmodels.api as sm
import datetime as dt

from statsmodels.tsa.stattools import adfuller

# current idea now is cluster --> use johansen test to test for cointegration --> minimise portmanteau stat --> parametric threshold
# or cluster --> engle granger test --> pairs with kalman filter --> parametric threshold

etfs_dict = {
	# (1)
	'alternative': ['DALT'],
	# (219)
	'bonds': ['ADFI','AFIF','AGG','AGGY','AGZ','AGZD','ANGL','ARCM','AVIG','AWTM','BIV','BKAG','BKHY','BKSB','BLV','BND','BNDC','BSJL','BSJM','BSJN','BSJP','BSJQ','BSJR','BSJS','BSV','BTC','CEFS','CLTL','CMBS','CORP','DEED','DFHY','DIAL','DWFI','EDV','ESCR','ESHY','FALN','FCOR','FDHY','FFIU','FIGB','FISR','FLBL','FLCO','FLDR','FLGV','FLHY','FLOT','FLRN','FLRT','FLTB','FLTR','FSEC','FTSD','FTSL','FTSM','FWDB','GBF','GBIL','GCOR','GHYB','GHYG','GNMA','GOVT','GOVZ','GSIG','GSY','GTIP','GVI','HCRB','HOLD','HSRT','HTAB','HYBB','HYD','HYDB','HYDW','HYG','HYGH','HYGV','HYHG','HYIN','HYLB','HYLS','HYLV','HYMB','HYMU','HYS','HYTR','HYUP','HYXF','HYXU','HYZD','IBHA','IBHB','IBHC','IBHD','IBHE','ICSH','IEF','IEI','IG','IGBH','IGEB','IGHG','IGIB','IGLB','IGSB','IHY','IHYF','IIGD','IIGV','ILTB','ISTB','JAGG','JIGB','JMBS','JNK','JPHY','JPST','JSCP','KDFI','KORP','LDSF','LDUR','LGOV','LMBS','LQDH','LQDI','LSST','LTPZ','MBB','MBBB','MBSD','MIG','MINT','MTGP','NEAR','NFLT','NUAG','NUBD','NUHY','NUSA','OPER','OVT','PBND','PBTP','PHYL','PULS','QLTA','RAVI','RBND','RDFI','RINF','SCHI','SCHJ','SCHO','SCHP','SCHQ','SCHR','SCHZ','SHAG','SHV','SHY','SHYD','SHYG','SHYL','SJNK','SKOR','SLQD','SNLN','SPAB','SPBO','SPHY','SPIB','SPSB','SPTI','SPXB','SRLN','STIP','STPZ','SUSB','SUSC','TBJL','TDTF','TDTT','TFJL','TFLO','TFLT','TGIF','THY','TIP','TIPZ','TLH','TLT','ULTR','USFR','USHY','USI','USIG','USTB','VABS','VALT','VCIT','VCLT','VCSH','VGIT','VGLT','VGSH','VMBS','VNLA','VPC','VRP','VTC','VTIP','WBII','WINC','ZROZ'],
	# (18)
	'commodities_broad_basket': ['BCD','BCI','BCM','CCRV','COM','COMB','COMT','DBC','DJP','FTGC','GCC','GSG','GSP','JJS','JO','RJI','SDCI','UCIB'],
	# (10)
	'communications': ['EWCO','FCOM','FIVG','IWFH','IXP','IYZ','JHCS','VOX','XLC','XTL'],
	# (25)
	'consumer_discretionary': ['AWAY','BEDZ','BETZ','BJK','CARZ','EATZ','FDIS','FTXD','FXD','IBUY','ITB','IYC','NERD','ONLN','PBS','PEJ','PEZ','PSCD','RCD','RTH','RXI','VCR','XHB','XLY','XRT'],
	# (12)
	'consumer_staples': ['FSTA','FTXG','FXG','IECS','IYK','KXI','PBJ','PSCC','PSL','RHS','VDC','XLP'],
	# (11)
	'currencies': ['CEW','CYB','FXA','FXB','FXC','FXE','FXF','FXY','UDN','USDU','UUP'],
	# (14)
	'derivatives': ['CWB','DBMF','FCVT','FMF','FUT','ICVT','KMLM','MOM','OVB','OVF','OVL','OVM','OVS','WTMF'],
	# (169)
	'developed_markets': ['AFK','AGT','ARGT','AVMU','BBEU','BBJP','BDRY','BOTZ','BWX','BWZ','CMF','DAX','DBEU','DBEZ','DBGR','DBJP','DFE','DRIV','DUDE','DXGE','ECH','EDEN','EFNL','EGPT','EIDO','EIRL','EIS','ENOR','ENZL','EPHE','EPOL','EPU','ERTH','ERUS','EUCG','EUDG','EURZ','EWA','EWC','EWD','EWG','EWGS','EWI','EWJ','EWJE','EWJV','EWK','EWL','EWM','EWN','EWO','EWP','EWQ','EWS','EWU','EWUS','EWW','EWY','EZA','EZU','FAN','FCEF','FDD','FEUL','FEZ','FGM','FIEE','FKU','FLIA','FLMI','FMB','FNI','FPXE','FSZ','GAA','GMOM','GREK','GRID','GSEU','GSJY','GXF','GYLD','HEWG','HEWU','HEWW','HEZU','HJPX','HMOP','IBMO','IBND','ICLN','ICOL','IDX','IEUS','IEV','IGOV','INMU','ISHG','ISRA','ITM','IZRL','JETS','JMUB','JPN','JPXN','KWT','LDRS','MBND','MCEF','MCRO','MEAR','MINN','MJ','MJJ','MLN','MMIN','MMIT','MSOS','MUB','MUNI','MUST','NGE','NLR','NYF','OEUR','PAWZ','PBD','PBW','PEX','PGAL','PLAT','QCLN','QPX','RAAX','RIGS','RLY','ROMO','RSX','RSXJ','RTAI','RVNU','SCJ','SHM','SMB','SMEZ','SMMU','SMOG','SPEU','SUB','TAN','TAXF','TFIV','THCX','THD','TOKE','TRND','TUR','UAE','VGK','VICE','VNM','VOO','VTEB','WIP','XMPT','ZCAN','ZDEU','ZGBR','ZJPN'],
	# (142)
	'emerging_markets': ['AAXJ','ADIV','ADRE','AFTY','AIA','ASEA','ASHR','ASHS','BICK','BKEM','BKF','BRF','BSAE','BSBE','BSCE','BSDE','CBON','CEMB','CEY','CHB','CHIE','CHIH','CHII','CHIM','CHIQ','CHIS','CHIX','CN','CNXT','CQQQ','CXSE','DBEM','DGS','DMRE','DVYA','DVYE','EAPR','EBND','ECNS','ECON','ECOW','EDIV','EDOG','EELV','EEM','EEMA','EEMD','EEMO','EEMS','EEMX','EFIX','EJAN','ELD','EMAG','EMB','EMBD','EMBH','EMCB','EMDV','EMFM','EMGF','EMHC','EMHY','EMIF','EMLC','EMQQ','EMSG','EMSH','EMTL','EMXC','EPI','EPP','ESEB','EWEB','EWX','EWZ','EWZS','EYLD','FCA','FEM','FEMS','FLN','FLQE','FM','FNDE','FPA','GEM','GLCN','GLIN','GMF','GSEE','GXC','HAUZ','HYEM','IEMG','ILF','INCO','INDA','INDY','IPAC','ISEM','IXSE','JEMA','JHEM','JPEM','JPMB','KALL','KBA','KBUY','KEMQ','KEMX','KFVG','KFYP','KGRN','KMED','KSTR','KURE','KWEB','LEMB','MCHI','NFTY','PBEE','PGJ','PIE','PIN','PXH','QEMM','RESE','RFEM','RNEM','ROAM','SCHE','SMIN','SOVB','SPEM','TLTE','UEVM','VPL','VWO','VWOB','XCEM','ZHOK'],
	# (40)
	'energy': ['ACES','AMJ','AMLP','AMND','AMUB','AMZA','ATMP','BMLP','CNRG','CRAK','EINC','EMLP','ENFR','FCG','FENY','FILL','FRAK','FXN','IEO','IEZ','IMLP','IXC','IYE','MLPB','MLPO','MLPX','OIH','PSCE','PXE','PXI','PXJ','PYPE','RYE','TPYP','UMI','USAI','VDE','XES','XLE','XOP'],
	# (48)
	'equities': ['AOA','AOK','AOM','AOR','ASPY','CLIX','CVY','DBEH','DIVA','DWSH','DYHG','FDIV','FFSG','FLYT','FTLS','GAL','HIPS','HNDL','HTUS','INKM','IPFF','IYLD','MDIV','OCIO','PCEF','PFF','PFFA','PFFD','PFLD','PFXF','PGF','PGX','PHDG','PSK','PSMB','PSMC','PSMG','PSMM','PWS','QLS','QPT','RISN','RPAR','TACE','TEGS','USHG','VAMO','YLD'],
	# (630)
	'factors': ['ABEQ','ACSG','ACSI','ACTV','ACWX','AESR','AFLG','AFSM','AIEQ','ALTL','AMOM','ARKK','ARMR','AZAA','AZAJ','AZAL','AZAO','AZBA','AZBJ','AZBL','AZBO','BBIN','BBMC','BFOR','BIBL','BKIE','BKLC','BKMC','BKSE','BMAR','BMAY','BOB','BUFF','BUL','CACG','CALF','CAPE','CATH','CDL','CEFA','CFA','CFCV','CHGX','CID','CIL','CIZ','CLRG','COWZ','CSA','CSB','CSD','CSF','CSM','CSML','CWI','CWS','CZA','DALI','DBAW','DBEF','DBJA','DBLV','DBOC','DDIV','DDLS','DDWM','DEEF','DEEP','DEF','DEMZ','DES','DEUS','DFAI','DFAU','DFNV','DGRO','DGRW','DHS','DIA','DIM','DINT','DIV','DJD','DLN','DLS','DMDV','DMRI','DMRL','DMRM','DMRS','DMXF','DNL','DOL','DON','DSI','DSJA','DSOC','DSTL','DURA','DUSA','DVOL','DWAS','DWAT','DWCR','DWM','DWMC','DWPP','DWX','EASG','ECOZ','EDOW','EEH','EES','EFA','EFAD','EFAV','EFAX','EFG','EFIV','EFV','EGIS','EPS','EQAL','EQL','EQRR','EQWL','ERM','ERSX','ESG','ESGA','ESML','ESNG','ETHO','EUSA','EWMC','EWSC','FAB','FAD','FBCG','FBCV','FBGX','FCPI','FDG','FDL','FDLO','FDM','FDMO','FDNI','FDRR','FDT','FDTS','FDVV','FEVR','FEX','FFTG','FGD','FGRO','FICS','FID','FIDI','FIVA','FLGE','FLQH','FLQL','FLQM','FLQS','FLV','FMAG','FMIL','FNDA','FNDB','FNDC','FNDF','FNDX','FNK','FNX','FNY','FPX','FPXI','FQAL','FRLG','FRTY','FTA','FTC','FTCS','FV','FVAL','FVC','FVD','FYC','FYLD','FYT','FYX','GBGR','GBLO','GLRY','GSEW','GSID','GSIE','GSLC','GSPY','GSSC','GSUS','GURU','GVAL','GVIP','GWX','HAIL','HAWX','HDAW','HDEF','HDMV','HDV','HFXI','HIPR','HLAL','HSCZ','HSMV','HUSV','IAPR','ICOW','IDHD','IDHQ','IDIV','IDLB','IDLV','IDMO','IDOG','IDV','IEFA','IFV','IHDG','IJAN','IJH','IJJ','IJK','IJR','IJS','IJT','ILCB','ILCG','ILCV','IMCB','IMCG','IMCV','IMOM','INTF','IPKW','IPO','IPOS','IQDE','IQDF','IQDG','IQDY','IQIN','IQLT','IQSI','IQSU','ISCB','ISCF','ISCG','ISCV','ISDX','ISMD','ISZE','ITOT','IUS','IUSG','IUSS','IUSV','IVAL','IVE','IVOG','IVOO','IVOV','IVV','IVW','IWB','IWC','IWD','IWF','IWL','IWM','IWN','IWO','IWP','IWR','IWS','IWV','IWX','IWY','IXUS','IYY','JDIV','JHMD','JHML','JHMM','JMIN','JMOM','JPIN','JPME','JPSE','JPUS','JQUA','JSMD','JSML','JUST','JVAL','KAPR','KJAN','KJUL','KLCD','KNG','KOMP','KSCD','KVLE','LCG','LCR','LCTU','LGH','LGLV','LRGE','LSAF','LVHD','LVHI','LVOL','LYFE','MAGA','MDY','MDYG','MDYV','MFDX','MFMS','MGC','MGK','MGMT','MGV','MID','MIDF','MMTM','MOAT','MOTI','MSVX','MTUM','MXDU','NAPR','NIFE','NJAN','NOBL','NTSX','NULC','NVQ','OEF','OMFL','OMFS','ONEO','ONEQ','ONEV','ONEY','OSCV','OUSA','OVLH','PALC','PAMC','PBDM','PBSM','PBUS','PDEV','PDN','PDP','PEY','PFM','PID','PIZ','PKW','PQIN','PRF','PRFZ','PSCW','PSCX','PSFD','PSFM','PSMD','PSMR','PTIN','PTLC','PTMC','PTNQ','PWB','PWC','PWV','PXF','QDEF','QDF','QDIV','QDYN','QEFA','QGRO','QINT','QMJ','QQC','QQD','QQEW','QQH','QQQ','QQQE','QQQJ','QQQM','QQQN','QQXT','QRFT','QSY','QUAL','QUS','RBIN','RBUS','RDIV','RDVY','REGL','RESP','RFDA','RFFC','RFG','RFV','RNLC','RNMC','RNSC','RODI','RODM','RORO','RPG','RPV','RSP','RVRS','RWGV','RWJ','RWK','RWL','RWVG','RYJ','RZG','RZV','SCHA','SCHB','SCHC','SCHD','SCHF','SCHG','SCHK','SCHM','SCHV','SCHX','SCZ','SDGA','SDOG','SDY','SECT','SENT','SFYF','SHE','SIXA','SIXH','SIXL','SIXS','SLT','SLY','SLYG','SLYV','SMCP','SMDV','SMLV','SNPE','SPDV','SPDW','SPGP','SPHB','SPHD','SPHQ','SPLG','SPLV','SPMD','SPMO','SPMV','SPQQ','SPSM','SPTM','SPVM','SPVU','SPXE','SPXN','SPXT','SPXV','SPXZ','SPY','SPYG','SPYV','SPYX','SQLV','SSLY','SSUS','STLG','STLV','STNC','STSB','SUSA','SUSL','SVAL','SVOL','SVXY','SYE','SYG','SYLD','SYUS','SYV','TAAG','TADS','TAEQ','TERM','TILT','TLTD','TMDV','TMFC','TPHD','TPIF','TPLC','TPSC','TRTY','TSJA','TSOC','TTAC','TTAI','TUSA','UIVM','ULVM','UMAR','UMAY','USEQ','USLB','USMC','USMF','USMV','USSG','USVM','UTRN','UVXY','VALQ','VB','VBK','VBR','VEA','VEGN','VETS','VEU','VFLQ','VFMF','VFMO','VFMV','VFQY','VFVA','VIG','VIGI','VIIXF','VIOG','VIOO','VIOV','VIRS','VIXM','VLU','VO','VOE','VONE','VONG','VONV','VOOG','VOOV','VOT','VPOP','VRAI','VSDA','VSL','VSMV','VSS','VTHR','VTI','VTRN','VTV','VTWG','VTWO','VTWV','VUG','VUSE','VV','VXF','VXUS','VYM','VYMI','WBIE','WBIF','WBIG','WBIL','WBIN','WBIT','WBIY','WIL','WOMN','WWJD','XDIV','XDQQ','XDSQ','XJH','XJR','XLG','XLSR','XMHQ','XMLV','XMMO','XMVM','XOUT','XRLV','XSHD','XSHQ','XSLV','XSMO','XSVM','XVOL','XVV','XVZ','YLDE','YYY','ZIVZF'],
	# (29)
	'financials': ['BDCZ','BIZD','DFNL','EUFN','FNCL','FXO','IAI','IAK','IAT','IEFN','IXG','IYF','IYG','JHMF','KBE','KBWB','KBWD','KBWP','KBWR','KCE','KIE','KRE','LEND','PFI','PSCF','QABA','RYF','VFH','XLF'],
	# (39)
	'health_care': ['AGNG','ARKG','BBC','BBH','BBP','BMED','BTEC','CNCR','EDOC','FBT','FHLC','FTXH','FXH','HART','HLGE','HTEC','IBB','IBBJ','IBBQ','IEHS','IEIH','IHE','IHF','IHI','IXJ','IYH','PBE','PJP','PPH','PSCH','PTH','RYH','SBIO','VHT','XBI','XHE','XHS','XLV','XPH'],
	# (20)
	'industrials': ['AIRR','EVX','EXI','FIDU','FLM','FXR','ITA','IYJ','IYT','JOYY','KARS','PKB','PPA','PSCI','RGI','ROKT','VIS','XAR','XLI','XTN'],
	# (44)
	'materials': ['AQWA','CGW','COPX','CUT','EBLU','FIW','FMAT','FTAG','FTRI','FXZ','GDX','GDXJ','GNR','GOAU','GOEX','GRES','GUNR','HAP','IGE','IYM','JGLD','LIT','MOO','MXI','NANR','PHO','PIO','PSCM','PYZ','REMX','RING','RTM','SGDJ','SGDM','SIL','SILJ','SLVP','URA','URNM','VAW','VEGI','WOOD','XLB','XME'],
	# (35)
	'real_estate': ['BBRE','DRW','EWRE','FFR','FPRO','FREL','FRI','GQRE','ICF','IFGL','INDS','IYR','KBWY','MORT','NETL','NURE','OLD','PPTY','PSR','RDOG','REIT','REM','REZ','ROOF','RWO','RWR','RWX','SCHH','SRET','SRVR','USRT','VNQ','VNQI','WPS','XLRE'],
	# (66)
	'technology': ['AIQ','ARKF','ARKQ','ARKW','BLCN','BLOK','BTEK','CCON','CIBR','CLOU','DTEC','EKAR','ESPO','FDN','FINX','FITE','FNGS','FTEC','FXL','GAMR','GINN','HACK','IETC','IGM','IGV','IRBO','ITEQ','IXN','IYW','JHMT','KOIN','LRNZ','MOON','NXTG','PNQI','PRNT','PSCT','PSI','PSJ','PTF','PXQ','QTEC','QTUM','ROBT','RYT','SKYY','SMH','SOXQ','SOXX','TDIV','TDV','TECB','THNQ','TPAY','VCAR','VCLO','VFIN','VGT','WCLD','WFH','WUGI','XLK','XNTK','XSD','XSW','XWEB'],
	# (186)
	'trading': ['AGQ','BIB','BIS','BNKD','BNKU','BOIL','BRZU','BZQ','CHAD','CHAU','CLDL','CLDS','CROC','CURE','DDG','DDM','DFEN','DFVL','DFVS','DGLDF','DGP','DGZ','DIG','DOG','DPST','DRIP','DRN','DRV','DSLVF','DUG','DUSL','DUST','DXD','DZZ','EDC','EDZ','EET','EEV','EFO','EFU','EFZ','EMTY','EPV','ERX','ERY','EUFX','EUM','EUO','EURL','EWV','EZJ','FAS','FAZ','FNGD','FNGO','FNGU','FXP','GDXD','GDXU','GLL','GUSH','HDGE','HIBL','HIBS','INDL','IWDL','IWFL','IWML','JDST','JNUG','KOLD','KORU','LABD','LABU','LBJ','LTL','MEXX','MIDU','MJO','MTUL','MVV','MYY','MZZ','NAIL','NUGT','PFFL','PILL','PSQ','PST','QID','QLD','QULL','REK','RETL','REW','ROM','RUSL','RWM','RXD','RXL','SAA','SBB','SBM','SCC','SCDL','SCO','SDD','SDOW','SDP','SDS','SEF','SH','SIJ','SJB','SKF','SKYU','SMDD','SMN','SOXL','SOXS','SPDN','SPUU','SPXL','SPXS','SPXU','SQQQ','SRS','SRTY','SSG','SSO','SZK','TBF','TBT','TBX','TECL','TECS','TMF','TMV','TNA','TPOR','TQQQ','TTT','TVIXF','TWM','TYD','TYO','TZA','UBOT','UBR','UBT','UCC','UCO','UCYB','UDOW','UGAZF','UGE','UGL','UGLDF','UJB','ULE','UMDD','UPRO','UPV','UPW','URE','URTY','USD','USLVF','USML','UST','UTSL','UWM','UXI','UYG','UYM','VIXY','VXX','WANT','WEBL','WEBS','XPP','YANG','YCL','YCS','YINN','YXI'],
	# (17)
	'utilities': ['ECLN','FUTY','FXU','GII','GLIF','IDU','IGF','INFR','JXI','NFRA','PSCU','PUI','RYU','SIMS','TOLZ','VPU','XLU']
}

class Debugger:
	def __init__(self, logger=print):
		self.logger = logger

	def _log(self, *args):
		try:
			display(*args)
		except:
			self.logger(*args)

	def display_side_by_side(self, df_list, caption=''):
		try:
			from IPython.display import display_html
			html_str = ''
			for i in range(len(df_list)):
				df_styler = df_list[i].style.set_table_attributes("style='display:inline; vertical-align: top; margin-right:10px'")
				if len(caption)>0:
					df_styler = df_styler.set_caption(caption+str(i))
				html_str += df_styler._repr_html_()
			display_html(html_str, raw=True)
		except:
			pass

class QCUtils(Debugger):
	def __init__(self):
		try:
			super().__init__(self.Log)
		except:
			super().__init__()
		self.qb = QuantBook()

	def symbol_helper(self, id):
		return self.qb.Symbol(id).Value

	def get_data(self, tickers, start_date, end_date, resolution='D'):
		resolutions = { 'D': Resolution.Daily, 'H': Resolution.Hour, 'M': Resolution.Minute }
		for ticker in tickers:
			symbol = self.qb.AddEquity(ticker, resolutions[resolution]).Symbol # same as using ticker itself
		self.raw_history = self.qb.History(self.qb.Securities.Keys, start_date, end_date, resolutions[resolution])
		self.df = self.raw_history['close'].unstack(level=0)
		self.volume_df = self.raw_history['volume'].unstack(level=0)
		return self.df, self.volume_df


# note alpaca's data is dog shit because its only from one exchange --> meaning it excludes a lot of data completely
class AlpacaUtils(Debugger):
	def __init__(self):
		super().__init__()
	
	def get_data(self, tickers, start_date, end_date, resolution):
		self._log('Is resolution and start and end date as pulled data agreeable?')

		raw_dfs = []

		directory = './alpaca_data/bars/day'
		close_df, volume_df = None, None
		tickers_set = set(tickers)
		max_n1, max_n2 = 0, 0
		no_data_found = []

		for i, filename in enumerate(os.listdir(directory)):
			symbol = '.'.join(filename.split('.')[:-1])
			
			if symbol in tickers_set:
				df = pd.read_csv(directory+'/'+filename, index_col='timestamp')
				raw_dfs.append(df)
				
				close = df[['close']].rename(columns={'close': symbol})
				volume = df[['volume']].rename(columns={'volume': symbol})

				max_n1 = max(len(close), max_n1)
				max_n2 = max(len(volume), max_n2)

				if i == 0:
					close_df = close
					volume_df = volume
				else:
					close_df = close_df.join(close, how='outer')
					volume_df = volume_df.join(volume, how='outer')
				
				assert max_n1 == close_df.shape[0]
				assert max_n2 == volume_df.shape[0]

			else:
				no_data_found.append(symbol)
		
		self._log(no_data_found)

		return close_df, volume_df


#cell 1
class DataPipeline(AlpacaUtils):
	def __init__(self, tickers, start_date, validation_start_date, testing_start_date, end_date, resolution='D'):
		super().__init__()
		self.start_date = dt.datetime(*start_date)
		self.end_date = dt.datetime(*end_date)
		self.validation_start_date = dt.datetime(*validation_start_date)
		self.testing_start_date = dt.datetime(*testing_start_date)
		self.resolution = resolution
		self.original_tickers = tickers

		self.df, self.volume_df = self.get_data(self.original_tickers, self.start_date, self.end_date, self.resolution) # close price

	def preprocess_and_split_data(self, min_avg_volume=10000, min_avg_price=5, limit=7, percent=0.1):
		self.raw_training_df = self.df.loc[self.df.index < self.validation_start_date]
		self.validation_df = self.df.loc[(self.df.index >= self.validation_start_date) & (self.df.index < self.testing_start_date)]
		self.testing_df = self.df.loc[self.df.index >= self.testing_start_date]
		self.training_and_validation_df = self.df.loc[self.df.index < self.testing_start_date]

		self.raw_training_volume_df = self.volume_df.loc[self.volume_df.index < self.validation_start_date]

		# process shit only after splitting! doing before will introduce 
		# 1. survivorship bias due to removing bad cases
		# 2. look-ahead bias by removing stocks that became shit in future

		# filter by volume and price
		filtered_by_volume = self.raw_training_df.loc[:, self.raw_training_volume_df.mean() > min_avg_volume]
		filtered_by_price = filtered_by_volume.loc[:, self.raw_training_df.mean() > min_avg_price]

		# coordinate start timings for training data
		self.training_df = self.coordinate_start_timings(filtered_by_price, limit, percent) 

		self._log(f'{len(self.original_tickers)} to {self.training_df.shape[1]} tickers')
		self._log(f'{self.training_df.shape[0]} + {self.validation_df.shape[0]} + {self.testing_df.shape[0]} ({self.training_and_validation_df.shape[0]})') 

		return self.training_df, self.validation_df, self.testing_df, self.training_and_validation_df

	def coordinate_start_timings(self, df, limit, percent):
		# need to coordinate start timings for etfs because some were created later
		old_num_rows = df.shape[0]
		dropped_na_columns = self._ffill_and_dropna(df, limit=limit, thresh=int((1-percent)*old_num_rows)) 

		idx_to_start = df.notnull().all(axis=1).argmax() # first common non na value

		self._log(f'Df new start date {df.index[idx_to_start]}, removed first {idx_to_start} or {idx_to_start/old_num_rows*100:.2f}% rows')

		return df.iloc[idx_to_start:]

	def _ffill_and_dropna(self, df, limit=None, thresh=0):
		# returns na columns
		old_num_cols = df.shape[1]
		df.fillna(method='ffill', inplace=True, limit=limit)

		na_columns = df.isna().any()
		df.dropna(axis='columns', inplace=True, thresh=thresh)

		self._log('##### FFILL AND DROP NA #####')
		self._log(f'{old_num_cols} to {df.shape[1]} columns - {old_num_cols - df.shape[1]} NA columns dropped')
		self._log(f'Dropped [{df.columns[na_columns].to_list()}] columns')
		self._log('#############################')
		self._log('')

		return na_columns


#cell 2
class ClusterPipeline(Debugger):
	def __init__(self, pca_factors=15, min_samples=3):
		super().__init__()
		self.pca_factors = pca_factors
		self.min_samples = min_samples

	def find_clusters(self, df):
		R = df.pct_change().iloc[1:, :] # here the columns of R are the different observations.
		assert self._ffill_and_dropna(R, 'R', 10).shape[1] == 0 # avoid any stocks with missing returns
		norm_R = (R - R.mean()) / R.std()
		assert self._ffill_and_dropna(norm_R, 'Norm R', 10) == 0 # avoid any stocks with missing returns

		pca = PCA()
		pca.fit(norm_R.T) # use returns as columns and stocks as rows
		pca_data = pca.transform(norm_R.T) # get PCA coordinates for scaled_data

		X = pca_data[:, :self.pca_factors]
		X = pd.DataFrame(X, columns=['PC'+str(i) for i in range(1, self.pca_factors+1)], index=norm_R.columns)

		self._log(f'{np.sum(pca.explained_variance_ratio_[:self.pca_factors] * 100)}% of variance - {self.pca_factors} components')

		optics_model = OPTICS(min_samples=self.min_samples)
		# min_samples parameter -> min number of samples required to form a dense region
		# xi parameter -> max distance between two samples to be considered as a neighborhood
		# min_cluster_size -> min size of a dense region to be considered as a cluster

		clustering = optics_model.fit(X)
		clusters = []
		for i in range(len(set(optics_model.labels_))-1):
			cluster = list(X[optics_model.labels_==i].index)
			clusters.append(cluster)

		self.plot_clusters(pca, optics_model, X)

		return clusters

	def plot_clusters(self, pca, optics_model, X):
		if not self.display_graphs:
			return
		# PCA plot
		per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
		labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
		plt.figure(figsize=(5, 3))
		plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
		plt.ylabel('Percentage of Explained Variance')
		plt.xlabel('Principal Component')
		plt.title('Scree Plot')
		plt.show()

		# Cluster plots
		space = np.arange(len(X))
		reachability = optics_model.reachability_[optics_model.ordering_]
		labels = optics_model.labels_[optics_model.ordering_]

		plt.figure(figsize=(12, 3))
		G = gridspec.GridSpec(1, 3)
		ax1 = plt.subplot(G[0, :2])
		ax2 = plt.subplot(G[0, -1])

		colors = ['r', 'g','b','c','y','m', 'coral', 'darkgreen', 'crimson', 'darkblue', 'ivory', 'khaki', 'r', 'g','b','c','y','m', 'coral', 'darkgreen', 'crimson', 'darkblue', 'ivory', 'khaki', 'r', 'g','b','c','y','m']

		assert len(set(labels)) <= len(colors)

		for i, color in enumerate(colors):
			Xk = space[labels == i]
			Rk = reachability[labels == i]
			ax1.plot(Xk, Rk, color, alpha = 0.3, marker='.')
			ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha = 0.3)
			ax1.plot(space, np.full_like(space, 2., dtype = float), 'k-', alpha = 0.5)
			ax1.plot(space, np.full_like(space, 0.5, dtype = float), 'k-.', alpha = 0.5)
			ax1.set_ylabel('Reachability Distance')
			ax1.set_title('Reachability Plot')

		# Plotting the OPTICS Clustering
		for i, color in enumerate(colors):
			Xk = X[optics_model.labels_ == i]
			ax2.plot(Xk.iloc[:, 0], Xk.iloc[:, 1], color, alpha = 0.3, marker='.')
			ax2.plot(X.iloc[optics_model.labels_ == -1, 0], X.iloc[optics_model.labels_ == -1, 1],'k+', alpha = 0.1)
			ax2.set_title('OPTICS Clustering')

		plt.tight_layout()
		plt.show()

		self.display_side_by_side([pd.DataFrame({'Name': map(self.symbol_helper, cluster)}) for cluster in clusters], 'Cluster ')




#cell 3
class PortfolioPipeline(Debugger):
	def __init__(self, p_value_threshold=0.01, max_half_life=60):
		super().__init__()
	 	# stationarity tests
		self.p_value_threshold = p_value_threshold
		self.min_half_life, self.max_half_life = 0, max_half_life # I want 1 week half life
		self.avg_cross_period_threshold = int(self.max_half_life * 0.75) # i'll just make it less strict for now

	def estimate_long_run_short_run_relationships(self, y, x):
		assert isinstance(y, pd.Series), 'Input series y should be of type pd.Series'
		assert isinstance(x, pd.Series), 'Input series x should be of type pd.Series'
		assert sum(y.isnull()) == 0, 'Input series y has nan-values. Unhandled case.'
		assert sum(x.isnull()) == 0, 'Input series x has nan-values. Unhandled case.'
		assert y.index.equals(x.index), 'The two input series y and x do not have the same index.'
		
		x = sm.add_constant(x)
		long_run_ols = sm.OLS(y, x)
		long_run_ols_fit = long_run_ols.fit()
		
		c, gamma = long_run_ols_fit.params
		z = long_run_ols_fit.resid

		short_run_ols = OLS(y.diff().iloc[1:], (z.shift().iloc[1:]))
		short_run_ols_fit = short_run_ols.fit()
		
		alpha = short_run_ols_fit.params[0]
				
		return c, gamma, alpha, z

	def engle_granger_two_step_cointegration_test(self, y, x):
		assert isinstance(y, pd.Series), 'Input series y should be of type pd.Series'
		assert isinstance(x, pd.Series), 'Input series x should be of type pd.Series'
		assert sum(y.isnull()) == 0, 'Input series y has nan-values. Unhandled case.'
		assert sum(x.isnull()) == 0, 'Input series x has nan-values. Unhandled case.'
		assert y.index.equals(x.index), 'The two input series y and x do not have the same index.'
		
		c, gamma, alpha, z = self.estimate_long_run_short_run_relationships(y, x)
		
		# NOTE: The p-value returned by the adfuller function assumes we do not estimate z first, but test 
		# stationarity of an unestimated series directly. This assumption should have limited effect for high N, 
		# so for the purposes of this course this p-value can be used for the EG-test. Critical values taking 
		# this into account more accurately are provided in e.g. McKinnon (1990) and Engle & Yoo (1987).
		
		adfstat, pvalue, usedlag, nobs, crit_values = adfuller(z, maxlag=1, autolag=None)
	
		return c, gamma, alpha, z, adfstat, pvalue

	def stationarity_check(self, price_series1, price_series2):
		constant, beta, alpha, residual, adfstat, p_value = self.engle_granger_two_step_cointegration_test(price_series1, price_series2)

		H, half_life, avg_cross_period = None, None, None

		if not(p_value <= self.p_value_threshold):
			return 0, constant, beta, alpha, residual, p_value, H, half_life, avg_cross_period # first number is index failed to track failed count

		# Hurst Exponent
		H, c, _data = compute_Hc(residual)
		if H >= 0.5: # spread is not mean-reverting
			return 1, constant, beta, alpha, residual, p_value, H, half_life, avg_cross_period

		# Half-life - duration to mean-revert
		half_life = -np.log(2) / alpha
		if not(self.min_half_life <= half_life and half_life <= self.max_half_life):
			return 2, constant, beta, alpha, residual, p_value, H, half_life, avg_cross_period

		# Mean cross frequency
		resid = np.array(residual)
		total_crosses = ((resid[:-1] * resid[1:]) < 0).sum()
		avg_cross_period = len(price_series1) / total_crosses
		if avg_cross_period > self.avg_cross_period_threshold:
			return 3, constant, beta, alpha, residual, p_value, H, half_life, avg_cross_period
		
		assert abs(residual.mean()) <= 1e-9

		return -1, constant, beta, alpha, residual, p_value, H, half_life, avg_cross_period

	def try_validate_pair(self, failed_count, df_data, price_series1, price_series2):
		failed_idx, constant, beta, alpha, residual, p_value, H, half_life, avg_cross_period = self.stationarity_check(price_series1, price_series2)
		if failed_idx != -1:
			failed_count[failed_idx] += 1
			return

		if beta < 0:
			self._log(f'Found pair with negative beta - {cluster[i]} {cluster[j]}')
			return

		df_data['Stock1'].append(cluster[i])
		df_data['Stock2'].append(cluster[j])
		df_data['Beta'].append(beta)
		df_data['p'].append(p_value)
		df_data['H'].append(H)
		df_data['Half-life'].append(half_life)
		df_data['Avg zero cross period'].append(int(avg_cross_period))
		df_data['Cluster'].append(int(cluster_idx))

	def log_failed_count(self, failed_count):
		self._log(f'{failed_count[0]} failed cointegration test')
		self._log(f'{failed_count[1]} failed H exp criterion')
		self._log(f'{failed_count[2]} failed half-life criterion')
		self._log(f'{failed_count[3]} failed avg zero cross period criterion')

	def revalidate_pairs(self, filtered_validation_results):
		test_pairs_data = { 'Stock1': [], 'Stock2': [], 'Beta': [], 'p': [], 'H': [], 'Half-life': [], 'Avg zero cross period': [], 'Cluster': [] }
		failed_count = [0]*4
		old_num_pairs = len(filtered_validation_results)

		for pair_key in filtered_validation_results:
			stock1, stock2 = pair_key.split('-')

			pair_dict = results[pair_key]

			training_pair_df = pair_dict['training_pair_df']
			pair_df = pair_dict['pair_df']
			df = pd.concat([training_pair_df, pair_df])

			price_series1, price_series2 = df[stock1], df[stock2]
			self.try_validate_pair(failed_count, test_pairs_data, price_series1, price_series2)

		test_pairs = pd.DataFrame(test_pairs_data)
		new_num_pairs = test_pairs.shape[0]

		self._log(f'{new_num_pairs}/{old_num_pairs} ({new_num_pairs/old_num_pairs*100:.2f}%) passed stationary check')
		self.log_failed_count(failed_count)

		self._log(test_pairs)

		return test_pairs


	def find_pairs_from_clusters(self, df, clusters):
		cluster_pairs = []
		failed_count = [0]*4
		total_num_of_pairs = 0

		for cluster_idx, cluster in enumerate(clusters):
			n = len(cluster)
			num_of_pairs = n*(n-1)//2
			total_num_of_pairs += num_of_pairs

			cluster_data = { 'Stock1': [], 'Stock2': [], 'Beta': [], 'p': [], 'H': [], 'Half-life': [], 'Avg zero cross period': [], 'Cluster': [] }
			self._log(f'Testing {num_of_pairs} pairs in cluster {cluster_idx}')

			for i in range(n):
				for j in range(i+1, n):
					price_series1, price_series2 = df[cluster[i]], df[cluster[j]]
					self.try_validate_pair(failed_count, cluster_data, price_series1, price_series2)

			cluster_pairs.append(pd.DataFrame(cluster_data))

		self._log(f'Tested {total_num_of_pairs} pairs in total')
		self.log_failed_count(failed_count)

		if len(cluster_pairs) == 0:
			return None
		pairs = pd.concat(cluster_pairs, ignore_index=True)
		self._log(f'Found {pairs.shape[0]} pairs')
		self._log(pairs)

		return pairs




#cell 4
class BacktestPipeline(Debugger):
	def __init__(self, percent_margin_buffer=0.1):
		super().__init__()
		self.percent_margin_buffer = percent_margin_buffer

	def prepare_training_and_testing_df(self, training_df, testing_df, stock1, stock2, beta):
		training_pair_df = training_df.loc[:, [stock1, stock2]]
		pair_df = testing_df.loc[:, [stock1, stock2]]

		training_pair_df_spread = training_pair_df[stock1] - beta * training_pair_df[stock2]
		pair_df_spread = pair_df[stock1] - beta * pair_df[stock2]

		mean = np.mean(training_pair_df_spread)
		std = np.std(training_pair_df_spread)

		training_pair_df['z'] = (training_pair_df_spread - mean) / std
		pair_df['z'] = (pair_df_spread - mean) / std

		return training_pair_df, pair_df

	def test_backtest(self, training_and_validation_df, testing_df, test_pairs, validation_backtest_results, initial_capital=1000):
		test_backtest_results = {}

		for stock1, stock2, beta, p, H, half_life, avg_cross_period, cluster_idx in test_pairs.values:
			pair_dict = validation_backtest_results[stock1+'-'+stock2]

			entry_z_threshold = pair_dict['entry_z_threshold']
			exit_z_threshold = pair_dict['exit_z_threshold']

			# self._log(f'Simulating pair [{self.symbol_helper(stock1)}-{self.symbol_helper(stock2)}]')
			training_pair_df, pair_df = self.prepare_training_and_testing_df(training_and_validation_df, testing_df, stock1, stock2, beta)

			position, fees, margin = self.backtest(pair_df, stock1, stock2, beta, initial_capital, entry_z_threshold, exit_z_threshold)

			test_backtest_results[stock1+'-'+stock2] = {
				'training_pair_df': training_pair_df,
				'pair_df': pair_df,
				'beta': beta,
				'entry_z_threshold': entry_z_threshold,
				'exit_z_threshold': exit_z_threshold,
				'position': position,
				'margin': margin,
				'fees': fees
			}

		return test_backtest_results

	def validation_backtest(self, training_df, validation_df, pairs, initial_capital=1000):
		validation_backtest_results = {}

		for entry_z_threshold in np.linspace(1.0, 2.5, 5):
			for exit_z_threshold in np.linspace(0.0, 1.0, 4):
				for stock1, stock2, beta, p, H, half_life, avg_cross_period, cluster_idx in pairs.values:
					# self._log(f'Simulating pair [{self.symbol_helper(stock1)}-{self.symbol_helper(stock2)}]')
					training_pair_df, pair_df = self.prepare_training_and_testing_df(training_df, validation_df, stock1, stock2, beta)

					position, fees, margin = self.backtest(pair_df, stock1, stock2, beta, initial_capital, entry_z_threshold, exit_z_threshold)

					past_margin = validation_backtest_results[stock1+'-'+stock2].get('margin')
					if past_margin == None or margin[-1] > past_margin[-1]:
						validation_backtest_results[stock1+'-'+stock2] = {
							'training_pair_df': training_pair_df,
							'pair_df': pair_df,
							'beta': beta,
							'entry_z_threshold': entry_z_threshold,
							'exit_z_threshold': exit_z_threshold,
							'position': position,
							'margin': margin,
							'fees': fees
						}

		return validation_backtest_results

	def backtest(self, pair_df, stock1, stock2, beta, initial_capital, entry_z_threshold, exit_z_threshold):
		position = { stock1: [0], stock2: [0] }
		capital = initial_capital
		margin = [capital]
		fees = [(0, 0, 0)]

		for time, data_at_time in pair_df.iterrows():
			stock1_close = data_at_time[stock1]
			stock2_close = data_at_time[stock2]
			cur_z_spread = data_at_time['z']

			position_direction = np.sign(position[stock1][-1])

			stock1_shares, stock2_shares = 0, 0
			commission, slippage, short_rental = 0, 0, 0

			usable_capital = capital * (1-self.percent_margin_buffer)
			if position_direction == 0:
				if (cur_z_spread <= -entry_z_threshold or cur_z_spread >= entry_z_threshold):
					# adding the / 2 to avoid margin calls??? im p sure this isnt right tho
					if beta > 1:
						stock2_shares = min(np.floor(usable_capital / stock2_close / 2), np.floor(usable_capital / stock1_close * beta / 2))
						stock1_shares = np.ceil(stock2_shares / beta)
					else:
						stock1_shares = min(np.floor(usable_capital / stock1_close / 2), np.floor(usable_capital / stock2_close / beta / 2))
						stock2_shares = np.ceil(stock1_shares * beta)
					
					assert stock1_shares > 0
					assert stock2_shares > 0
						
					is_long = cur_z_spread <= -entry_z_threshold

					position[stock1].append(stock1_shares if is_long else -stock1_shares)
					position[stock2].append(-stock2_shares if is_long else stock2_shares)
					pos_stock1, pos_stock2 = position[stock1][-1], position[stock2][-1]

					portfolio_value = pos_stock1 * stock1_close + pos_stock2 * stock2_close
					commission = 0.0008 * (abs(pos_stock1) * stock1_close + abs(pos_stock2) * stock2_close)
					slippage = 0.0020 * (abs(pos_stock1) * stock1_close + abs(pos_stock2) * stock2_close)
					capital -= slippage + commission
					capital -= portfolio_value
					assert capital >= 0, (commission, slippage, portfolio_value, pos_stock1*stock1_close, pos_stock2*stock2_close, pos_stock1, pos_stock2, stock1_close, stock2_close, beta)
				else:
					position[stock1].append(0)
					position[stock2].append(0)
			else:
				short_rental = -position[stock2][-1] * stock2_close * 0.01/252 if position_direction > 0 else -position[stock1][-1] * stock1_close * 0.01/252
				capital -= short_rental
				if ((position_direction > 0 and cur_z_spread >= exit_z_threshold) or (position_direction < 0 and cur_z_spread <= -exit_z_threshold)):
					pos_stock1, pos_stock2 = position[stock1][-1], position[stock2][-1]
					portfolio_value = pos_stock1 * stock1_close + pos_stock2 * stock2_close
					commission = 0.0008 * (abs(pos_stock1) * stock1_close + abs(pos_stock2) * stock2_close)
					slippage = 0.0020 * (abs(pos_stock1) * stock1_close + abs(pos_stock2) * stock2_close)
					capital -= commission + slippage
					capital += portfolio_value

					position[stock1].append(0)
					position[stock2].append(0)
				else:
					position[stock1].append(position[stock1][-1])
					position[stock2].append(position[stock2][-1])
			
			pos_stock1, pos_stock2 = position[stock1][-1], position[stock2][-1]
			portfolio_value = pos_stock1 * stock1_close + pos_stock2 * stock2_close
			margin.append(capital + portfolio_value) # store margin if liquidated everything at point in time
			fees.append((commission, slippage, short_rental))

		return position, fees, margin

	def plot_pair_backtest(self, results, pair_key):
		stock1, stock2 = pair_key.split('-')

		pair_dict = results[pair_key]

		training_pair_df = pair_dict['training_pair_df']
		pair_df = pair_dict['pair_df']
		beta = pair_dict['beta']
		entry_z_threshold = pair_dict['entry_z_threshold']
		exit_z_threshold = pair_dict['exit_z_threshold']
		position	= pair_dict['position']
		margin	= pair_dict['margin']
		fees	= pair_dict['fees']

		self._log(f'[{self.symbol_helper(stock1)} {self.symbol_helper(stock2)}] Entry z threshold: {entry_z_threshold:.3f} Exit z threshold: {exit_z_threshold:.3f} Cum PnL: {(margin[-1]-margin[0]):.3f}')

		plt.figure(figsize =(12, 5))
		G = gridspec.GridSpec(2, 3)
		ax1 = plt.subplot(G[0, 0])
		ax2 = plt.subplot(G[0, 1])
		ax3 = plt.subplot(G[0, 2])
		ax4 = plt.subplot(G[1, 0])
		ax5 = plt.subplot(G[1, 1])
		ax6 = plt.subplot(G[1, 2])

		stock1_symbol = self.symbol_helper(stock1)
		stock2_symbol = self.symbol_helper(stock2)

		time = np.concatenate([training_pair_df.index, pair_df.index])

		# plot pairs individual price
		ax1.plot(time, np.concatenate([training_pair_df[stock1], pair_df[stock1]]), 'r', label=stock1_symbol)
		ax1.plot(time, np.concatenate([training_pair_df[stock2], pair_df[stock2]]), 'g', label=stock2_symbol)
		ax1.axvline(x=pair_df.index[0], ymin=0, ymax=1, linewidth=1, color='b')
		ax1.set_xlabel('Date')
		ax1.set_title('Close price pair comparison')
		ax1.legend()

		# plot z spread price
		ax2.plot(time, np.concatenate([training_pair_df['z'], pair_df['z']]), 'c', label=f'z = norm {stock1_symbol}-{beta:.2f}*{stock2_symbol}')
		ax2.axvline(x=pair_df.index[0], ymin=0, ymax=1, linewidth=1, color='b')

		pos_stock1 = np.array(position[stock1])[1:]
		pos_stock2 = np.array(position[stock2])[1:]
		np_margin = np.array(margin)

		long_indices = pos_stock1 > 0
		short_indices = pos_stock1 < 0
		ax2.plot(pair_df.index[long_indices], pair_df.loc[long_indices, 'z'], 'g.')
		ax2.plot(pair_df.index[short_indices], pair_df.loc[short_indices, 'z'], 'r.')
		ax2.axhline(y=entry_z_threshold, xmin=0, xmax=1, linewidth=1, color='m')
		ax2.axhline(y=-entry_z_threshold, xmin=0, xmax=1, linewidth=1, color='m')
		ax2.axhline(y=exit_z_threshold, xmin=0, xmax=1, linewidth=1, color='brown')
		ax2.axhline(y=-exit_z_threshold, xmin=0, xmax=1, linewidth=1, color='brown')
		ax2.set_xlabel('Date')
		ax2.set_title('z')
		ax2.legend()

		# margin, fees, and drawdown got one extra starting pt
		# plot cumulative pnl
		ax3.plot(np_margin)
		ax3.set_title('Margin')

		# plot fees
		np_fees = np.array(fees)
		cumsum_fees = np.cumsum(np_fees, axis=0)
		cumsum_total_fees = np.sum(cumsum_fees, axis=1)
		ax4.plot(cumsum_fees[:, 0], label='commission')
		ax4.plot(cumsum_fees[:, 1], label='slippage')
		ax4.plot(cumsum_fees[:, 2], label='short rental')
		ax4.plot(cumsum_total_fees, label='total')
		ax4.set_title('Fees')
		ax4.legend()

		# plot exposure 
		stock1_invested = pair_df[stock1] * pos_stock1
		stock2_invested = pair_df[stock2] * pos_stock2
		net_exposure = stock1_invested + stock2_invested
		abs_exposure = np.absolute(stock1_invested) + np.absolute(stock2_invested)
		ax5.plot(stock1_invested, label='stock1')
		ax5.plot(stock2_invested, label='stock2')
		ax5.plot(net_exposure, label='net exposure')
		ax5.plot(abs_exposure, label='abs exposure')
		ax5.set_title('Exposure')
		ax5.legend()

		# plot drawdown
		cumret = np_margin / np_margin[0] - 1
		highwatermark=np.zeros(cumret.shape)
		drawdown=np.zeros(cumret.shape)
		drawdownduration=np.zeros(cumret.shape)
		
		for t in np.arange(1, cumret.shape[0]):
			highwatermark[t]=np.maximum(highwatermark[t-1], cumret[t])
			drawdown[t]=(1+cumret[t])/(1+highwatermark[t])-1
			if drawdown[t]==0:
				drawdownduration[t]=0
			else:
				drawdownduration[t]=drawdownduration[t-1]+1
				
		maxDD, i=np.min(drawdown), np.argmin(drawdown) # drawdown < 0 always
		maxDDD=np.max(drawdownduration)
		
		ax6.plot(drawdown)
		ax6.set_title('Drawdown')

		plt.tight_layout()
		plt.show()
	
	



#cell 5
try:
	from datapipeline import DataPipeline
	from clusterpipeline import ClusterPipeline
	from portfoliopipeline import PortfolioPipeline
	from backtestpipeline import BacktestPipeline
except:
	pass

sectors_dict = {
	'final': ['DBCN UX9SXI5CAPNP', 'HEWG VNTW0AC8LAHX', 'GSJY W8L8B8ZCNXB9', 'ITF S96RH23DIAUD', 'EFO UD63CSAA26P1', 'UPV UM61FJMT8EHX', 'EFU TX34HT712KBP',  'EPV UDJVM3EN4QXX', 'DGZ U0K69ONGSDPH', 'DZZ U0J6TLAAPMJP'] #, 'GLL U85WJOCE24BP']
}

total_pairs_found = 0
total_pairs_validated = 0
final_results = pd.DataFrame({ 'Stock1': [], 'Stock2': [], 'Beta': [], 'p': [], 'H': [], 'Half-life': [], 'Avg zero cross period': [], 'Cluster': [] })

for sector in sectors_dict:
	display(f'Doing {sector} sector now')


	data_pipe = DataPipeline(sectors_dict[sector], (2018, 1, 1), (2020, 1, 1), (2022, 1, 1), (2023, 1, 1))
	cluster_pipe = ClusterPipeline() 
	portfolio_pipe = PortfolioPipeline()
	backtest_pipe = BacktestPipeline()


	training_df, validation_df, testing_df, training_and_validation_df = data_pipe.preprocess_and_split_data()

	clusters = [training_df.columns] if training_df.shape[1] < 15 else cluster_pipe.find_clusters(training_df)


	# get pairs for validation test
	validation_pairs = portfolio_pipe.find_pairs_from_clusters(training_df, clusters)
	if validation_pairs is None or validation_pairs.shape[0] == 0:
		display(f'No validated pairs in {sector} sector')
		continue

	total_pairs_found += validation_pairs.shape[0]

	initial_capital = 5000

	# validation backtest
	validation_backtest_results = backtest_pipe.validation_backtest(training_df, validation_df, validation_pairs, initial_capital=initial_capital)
	
	for pair_key in validation_backtest_results:
		backtest_pipe.plot_backtest_results(validation_backtest_results, pair_key)



	# filter results
	filtered_validation_backtest_results = {}

	for pair_key in validation_backtest_results:
		margin = validation_backtest_results[pair_key]['margin']
		if (margin[-1]-margin[0])/margin[0] >= 0.1:
			filtered_validation_backtest_results[pair_key] = validation_backtest_results[pair_key]

	num_successful = len(filtered_validation_backtest_results)
	num_total = len(validation_backtest_results)
	self._log(f'Sector {sector}: {num_successful}/{num_total} = {num_successful/num_total*100:.2f}% have +PnL')
	self._log(filtered_validation_backtest_results)

	if len(filtered_validation_pairs) == 0:
		display(f'No positive validated pairs in {sector} sector')
		continue

	total_pairs_validated += num_successful



	# get final test pairs
	test_pairs = portfolio_pipe.revalidate_pairs(training_and_validation_df, filtered_validation_pairs)



	# test backtest
	test_backtest_results = backtest_pipe.test_backtest(training_and_validation_df, testing_df, test_pairs, initial_capital=initial_capital)


	final_results = pd.concat([final_results, test_pairs], ignore_index=True)

display(final_results)
display(f'Final result for all sectors: Total pairs validated / total pairs found = {total_pairs_validated}/{total_pairs_found} = {total_pairs_validated/total_pairs_found*100:.2f}% pairs ')





