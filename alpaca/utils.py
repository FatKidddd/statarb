import pandas as pd 

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

def ffill_and_dropna(df, limit=None, thresh=0):
  # returns na columns
  old_num_cols = df.shape[1]
  df.fillna(method='ffill', inplace=True, limit=limit)

  na_columns = df.columns[df.isna().any()]
  df.dropna(axis='columns', inplace=True, thresh=thresh)

  display(f'{old_num_cols} to {df.shape[1]} columns - {old_num_cols - df.shape[1]} NA columns dropped')
  display(f'Dropped {na_columns.to_list()} columns')

  return na_columns

