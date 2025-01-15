## Funkcjonalność

- wizualizacja syntezy fouriera na podstawie zmiany spektrogramu audio wejściowego i odtworzenia audio na podstawie fazy audio wejściowego i zmienionego spektrogramu

## Teoria

Program składa się z paru faz:

Wyliczenie STFT - krótko-czasowa transformata Fourier'a sygnału audio:

$$\text{STFT}(t, f) = \sum_{n=0}^{N-1} x[n] \, w[n - tH] \, e^{-j \frac{2\pi}{N} fn}$$

Wyznaczenie fazy transformaty i zapisanie jej:

$$\theta(t, f) = \arg(\text{STFT}(t, f))$$

Wyznaczenie mel-spektrogramu. Zachowuje perceptualnie istotne częstotliwości. 

$$\text{MelSpec}(t, m) = \sum_{f} |\text{STFT}(t, f)|^2 \times \text{MelFilter}(f, m)$$

Zamiana na skalę decybelową celem dalszego zmniejszania nieistnotnych informacji, oraz łatwiejszego określenia min i max

$$\text{meldB}(t, m) = 10 \cdot \log_{10}(\text{MelSpec}(t, m)) - \max(\cdot)$$

Normalizacja do skali $[0, 255]$ celem zapisania w formacie png

$$\text{melNormalized}(t, m) = \frac{\text{meldB}(t, m) - \min(\text{meldB})}{\max(\text{meldB}) - \min(\text{meldB})} \times 255$$

Po przekształceniach następują operacje na obrazku spektrogramu oraz proces odwrotny - faktycznej syntezy fouriera:

$$\text{specDenorm}(t, m) = \left( \frac{\text{processedSpec}(t, m)}{255} \right) \times \max(\cdot) - \frac{\max(\cdot)}{2}$$

$$\text{specPower}(t, m) = 10^{\frac{\text{specDenorm}(t, m)}{10}}$$

$$\text{linearSpec}(t, f) = \text{InverseMel}(\text{specPower}(t, m))$$

W celu rekonstrukcji STFT następuje połączenie spektrogramu i uprzednio zapisanej fazy:

$$\text{stftReconstructed}(t, f) = \text{linearSpec}(t, f) \cdot e^{j \, \theta(t, f)}$$

$$x[n] = \sum_{t=0}^{T-1} \left[ \frac{1}{N} \sum_{k=0}^{N-1} \text{stftReconstructed}(t, f) \, e^{j \frac{2\pi}{N}f(n-tH)} \right] w[n-tH]$$