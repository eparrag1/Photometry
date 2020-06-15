# Photometry
Creating Supernova Light Curves

How to run:

This is to be run using Python 3. See Requirements.txt for the necessary packages.

Telescopes.csv contains the colour terms for the telescopes used to take the images. The default is for Las Cumbres Observatory and the Liverpool Telescope. If any of your images were taken by a telescope not contained in this file, you will need to add this in.

The filters this code works with are B,V,G,R,I,Z.

To execute the code run the script 'Class_PSF_calc.py'.

# Input

You will then be prompted for the following information:

*Directory:*
This is the location where you have stored your fits files.
You may wish to template subtract your images. If you are using template subtracted images, make sure to include in the directory the original images as well, and preface subtracted images as 'sub_image.fits'. You do not need to combine images from the same night and filter beforehand, but it is recommended.

*Cutoff:*
If you are using template subtraction for all images past a certain MJD, enter the MJD here. If you are only performing template subtraction for a few individual images, leave this blank.

*Groupid/Object name:*
The fits file headers will be checked to make sure the image is for the correct object. Enter the object name or groupid that would appear in the fits header to ensure any incorrect images which may have mixed in are removed.

*Galactic Coordinates:*
Enter the galactic coordinates of the object in format '21:00:20.930 -21:20:36.06'

*ja200 Coordinates:*
Enter the ja200 coordinates of the object in format '315.08720833 -21.34335'

# Output

The output will be an errobar scatter plot displaying the figure, 'Light_curve.png', and an excel spreadsheet containing the data, 'Photometry_data.xlsx'.

'Photometry_data.xlsx' has substituted numbers for the filters, as below:
B: 1, V:2, gp:3, ip:4, rp: 5, SDSS-I: 6, SDSS-R: 7, SDSS-Z: 8, SDSS-G: 9

# A quick note on catalogues
The catalogues in use 'II/336' for B and V filters, and 'II/349' for griz, from Vizier Pan-STARSS. It is possible that the location of your object is not covered by these catalogues. Currently nothing is implemented to handle this, therefore the workaround would be to refer to the Vizier database for appropriate catalogues and manually replace these in Class_PSF.py.
