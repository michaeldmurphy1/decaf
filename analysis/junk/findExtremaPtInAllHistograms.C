void findExtremaPtInAllHistograms() {
    // Open the ROOT file
    TFile *f = TFile::Open("/Users/sena/Grad School/Research/decaf/decaf/analysis/data/ElectronTrigEff/egammaEffi.txt_EGM2D-2016postVFP.root");

    // Check if the file is opened successfully
    if (!f || f->IsZombie()) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    // Loop over all keys in the file
    TIter next(f->GetListOfKeys());
    TKey *key;

    while ((key = (TKey*)next())) {
        // Get the object
        TObject *obj = key->ReadObj();
        // Check if it is a 2D histogram
        if (TH2F *h2 = dynamic_cast<TH2F*>(obj)) {
            double minPt = 1e6; // Start with a large number for minimum
            double maxPt = -1e6; // Start with a small number for maximum
            int minBinX, minBinY;
            int maxBinX, maxBinY;

            // Loop over all bins along X and Y axes
            for (int binX = 1; binX <= h2->GetNbinsX(); ++binX) {
                for (int binY = 1; binY <= h2->GetNbinsY(); ++binY) {
                    double content = h2->GetBinContent(binX, binY);

                    // Skip zero-content bins
                    if (content == 0) continue;

                    // Get the Y-axis (pT) value for the current bin
                    double pt = h2->GetYaxis()->GetBinCenter(binY);

                    // Update the minimum pT value if the current bin's pT is lower
                    if (pt < minPt) {
                        minPt = pt;
                        minBinX = binX;
                        minBinY = binY;
                    }
                    // Update the maximum pT value if the current bin's pT is higher
                    if (pt > maxPt) {
                        maxPt = pt;
                        maxBinX = binX;
                        maxBinY = binY;
                    }
                }
            }

            // Print the histogram name, lowest and highest non-zero pT values
            std::cout << "Histogram: " << h2->GetName() << std::endl;
            std::cout << "The lowest non-zero pT value is: " << minPt << " GeV" << std::endl;
            std::cout << "Located at bin (" << minBinX << ", " << minBinY << ")" << std::endl;
            std::cout << "The highest non-zero pT value is: " << maxPt << " GeV" << std::endl;
            std::cout << "Located at bin (" << maxBinX << ", " << maxBinY << ")\n" << std::endl;
        }
    }

    // Close the ROOT file
    f->Close();
}
