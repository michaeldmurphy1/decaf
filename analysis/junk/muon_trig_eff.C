void muon_trig_eff() {
    // Open the ROOT file
    TFile *f = TFile::Open("/Users/sena/Grad School/Research/decaf/decaf/analysis/data/ElectronTrigEff/egammaEffi.txt_EGM2D-2016postVFP.root");

    // Load the histogram
    TH2F *h = (TH2F*)f->Get("EGamma_EffMC2D;1");
    // echo the histogram name
    std::cout << h->GetName() << std::endl;
    

    // Assuming the histogram is not empty and properly loaded
    if(h) {
        // Initialize minimum pT value to a large number
        double minPt = 1e6; // 1e6 is just a placeholder for a very large number
        int minBinX, minBinY, minBinZ;

        // Loop over all bins along X and Y axes
        for (int binX = 1; binX <= h->GetNbinsX(); ++binX) {
            for (int binY = 1; binY <= h->GetNbinsY(); ++binY) {
                double content = h->GetBinContent(binX, binY);

                // Skip zero-content bins
                if (content == 0) continue;

                // Get the Y-axis (pT) value for the current bin
                double pt = h->GetYaxis()->GetBinCenter(binY);

                // Update the minimum pT value if the current bin's pT is lower
                if (pt < minPt) {
                    minPt = pt;
                    minBinX = binX;
                    minBinY = binY;
                }
            }
        }

        // Print the lowest non-zero pT value
        std::cout << "The lowest non-zero pT value is: " << minPt << " GeV" << std::endl;
        std::cout << "Located at bin (" << minBinX << ", " << minBinY << ")" << std::endl;
    } else {
        std::cout << "Histogram not found." << std::endl;
    }

    // Close the ROOT file
    f->Close();
}
