import urllib.request

urlbase = 'https://tendl.web.psi.ch/tendl_2019/proton_file/Ta/Ta181/tables/xs/'
urlbase = 'https://tendl.web.psi.ch/tendl_2019/neutron_file/Xe/'
targets = ['Xe129','Xe126','Xe128','Xe124','Xe130','Xe131','Xe132','Xe134','Xe136']
urlbridge = '/tables/residual/'




max_num = 3
counter = 0

for target in targets:

    #if counter == 1: break
    print('Grabbing data for {}'.format(target))

    targetA = target[2:]
    targetZ = 54 # Xenon

#    for Z in range(42,targetZ):
#        for A in range(90,int(targetA)):
    for Z in range(targetZ,targetZ+1):
        for A in range(90,int(targetA)+2):

             print('\t(A,Z) = ({},{})'.format(A,Z)) 

             tag = 'rp{:>03}{:>03}.tot'.format(Z,A)
             url = urlbase + target + urlbridge + tag
             print('\tOpening {}'.format(url))
             try:
                 infile = urllib.request.urlopen(url)
             except urllib.error.HTTPError:
                 continue
             output_file_tag = ''
            
             outfile_name = target+'/TENDL_{}_{:>03}{:>03}_xsec.txt'.format(target,Z,A)
             print('Writing to {}'.format(outfile_name))
             with open(outfile_name,'w') as outputfile:
                 for line in infile:
                    outputfile.write(line.decode('utf-8'))
                    #decoded_line = line.decode('utf-8')
                    #print(decoded_line)
    counter += 1                
    			

