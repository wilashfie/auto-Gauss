fn = 'iris_l2_20220331_174325_4204700137_raster_t000_r00000.fits'
d = iris_getwindata(fn,1403,ixrange=[2600,4600],/normalize)

; prep 
int_s = d.int 
int_prep = iris_prep_despike(int_s,mode='both')

d.int = int_prep  

save, d, filename = 'siiv_windat_f1.sav'

end