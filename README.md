# SINets2
some Improved Models base on SINet, as my final design in college

Results in COD10K compared to SINet and SINetv2:
-------------------------------------------------------------------   
Model Smeasure meanEm wFmeasure MAE   
SINet_Official 0.758 0.796 0.517 0.054    
SINet_Train0 0.766 0.790 0.522 0.055    
SINet_Train1 0.765 0.822 0.586 0.048    
SINetS 0.768 0.823 0.599 0.045    
SINetS_RFDCT_Dec1 0.775 0.836 0.611 0.044    
SINetS_RFDCTHalf2_Dec1 0.775 0.833 0.606 0.045    
SINetS_RFFFTHalf2_Dec1 0.774 0.833 0.612 0.044    
SINetS_RFDCTHalf_Dec3 0.774 0.837 0.613 0.043    
SINetS_RF2Half_Dec3 0.775 0.837 0.619 0.043    
SINetS_RF2FFTHalf_Dec3 0.775 0.840 0.614 0.043    
SINetS_RF2FFTHalf2_Dec3 0.774 0.834 0.609 0.045    

-------------------------------------------------------------------

Model Loss Epoch DataAug Val Smeasure meanEm wFmeasure MAE   
SINetv2(Official) SLoss 100 TRUE TRUE ***0.815 0.887 0.680 0.037***    
SINetv2 BCE 64 FALSE FALSE 0.794 0.840 0.624 0.041    
SI_Decs BCE 64 FALSE FALSE 0.797 0.847 0.639 0.040    
SINetv2 SLoss 64 FALSE FALSE 0.790 0.879 0.650 0.040    
SI_Decs SLoss 64 FALSE FALSE 0.805 0.886 0.679 0.037    
SINetv2 BCE 100 FALSE FALSE 0.790 0.842 0.626 0.041    
SI_Decs BCE 100 FALSE FALSE 0.799 0.856 0.643 0.039    
SINetv2 SLoss 100 FALSE FALSE 0.788 0.875 0.644 0.041    
SI_Decs SLoss 100 FALSE FALSE 0.800 0.884 0.675 0.037    
SINetv2 BCE 100 TRUE FALSE 0.813 0.836 0.626 0.040    
SI_Decs BCE 100 TRUE FALSE 0.822 0.856 0.662 0.036    
SINetv2 SLoss 100 TRUE FALSE ***0.810 0.882 0.672 0.038***    
SI_Decs SLoss 100 TRUE FALSE ***0.822 0.893 0.701 0.034***    
SI_MinDecs SLoss 100 TRUE FALSE ***0.822 0.896 0.702 0.034***     

------------------------------------------------------------------   
see https://github.com/DengPingFan/SINet and https://github.com/GewelsJI/SINet-V2

