# smart-feature data links

These links make it easy to share the exact versions of the files and avoid CDN caching issues on raw.githubusercontent.com.

## Historical FX Future & Options Combined.txt
- Blob (commit-pinned):
  https://github.com/forexnstock-beep/smart-feature/blob/3742e4a459300dc1d8e92294412cb998fb05d2d7/Historical%20FX%20Future%20%26%20Options%20Combined.txt
- Raw (commit-pinned):
  https://raw.githubusercontent.com/forexnstock-beep/smart-feature/3742e4a459300dc1d8e92294412cb998fb05d2d7/Historical%20FX%20Future%20%26%20Options%20Combined.txt
- Raw (tracks main):
  https://raw.githubusercontent.com/forexnstock-beep/smart-feature/main/Historical%20FX%20Future%20%26%20Options%20Combined.txt
- Raw (tracks main, cache-busted example):
  https://raw.githubusercontent.com/forexnstock-beep/smart-feature/main/Historical%20FX%20Future%20%26%20Options%20Combined.txt?_=1723384215

## Gold and Silver.txt
- Blob (commit-pinned):
  https://github.com/forexnstock-beep/smart-feature/blob/3742e4a459300dc1d8e92294412cb998fb05d2d7/Gold%20and%20Silver.txt
- Raw (commit-pinned):
  https://raw.githubusercontent.com/forexnstock-beep/smart-feature/3742e4a459300dc1d8e92294412cb998fb05d2d7/Gold%20and%20Silver.txt
- Raw (tracks main):
  https://raw.githubusercontent.com/forexnstock-beep/smart-feature/main/Gold%20and%20Silver.txt
- Raw (tracks main, cache-busted example):
  https://raw.githubusercontent.com/forexnstock-beep/smart-feature/main/Gold%20and%20Silver.txt?_=1723384215

## Verify and bypass caching

You can inspect caching headers and force a refresh with curl:

```
# Check cache age on the main-branch Raw URL
curl -I "https://raw.githubusercontent.com/forexnstock-beep/smart-feature/main/Historical%20FX%20Future%20%26%20Options%20Combined.txt"

# Compare with a cache-busted request (Age should reset)
curl -I "https://raw.githubusercontent.com/forexnstock-beep/smart-feature/main/Historical%20FX%20Future%20%26%20Options%20Combined.txt?_=1723384215"
```

Notes:
- Commit-pinned links are immutable and safest for sharing a known-good snapshot.
- Main-branch Raw URLs may be cached at different CDN edges briefly after updates; adding any query parameter (e.g., ?_=timestamp) typically refreshes them.
- If you ever run into UI truncation for very large files, prefer the Raw link or attach the file as a Release asset.
