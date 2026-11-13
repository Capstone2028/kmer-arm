param(
    [string]$AccessionFile = 'data/microviridae_accessions_20261111.txt',
    [string]$OutputFile = 'data/microviridae_records_20261111.gb',
    [int]$BatchSize = 150
)

$ErrorActionPreference = 'Stop'
if ($BatchSize -lt 1 -or $BatchSize -gt 200) {
    throw 'BatchSize must be between 1 and 200.'
}

$accessions = @(Get-Content -LiteralPath $AccessionFile | Where-Object { $_.Trim() })
if ($accessions.Count -eq 0) { throw "No accessions found in $AccessionFile" }

$partial = "$OutputFile.part"
$outputDirectory = Split-Path -Parent $OutputFile
if ($outputDirectory) { New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null }
if (Test-Path -LiteralPath $OutputFile) {
    throw "Output already exists: $OutputFile. Choose another OutputFile to avoid overwriting it."
}

$encoding = New-Object System.Text.UTF8Encoding($false)
$found = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
if (Test-Path -LiteralPath $partial) {
    $partialText = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $partial), $encoding)
    foreach ($match in [regex]::Matches($partialText, '(?m)^VERSION\s+(\S+)')) {
        [void]$found.Add($match.Groups[1].Value)
    }
}

$baseUri = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
$batchCount = [math]::Ceiling($accessions.Count / $BatchSize)
for ($start = 0; $start -lt $accessions.Count; $start += $BatchSize) {
    $end = [math]::Min($start + $BatchSize - 1, $accessions.Count - 1)
    $batch = @($accessions[$start..$end])
    $missing = @($batch | Where-Object { -not $found.Contains($_) })
    if ($missing.Count -eq 0) {
        Write-Output "Already have batch $([math]::Floor($start / $BatchSize) + 1)/$batchCount"
        continue
    }
    if ($missing.Count -ne $batch.Count) {
        throw "Partial batch found near record $start. Stop to avoid duplicate or incomplete records."
    }

    $body = @{
        db = 'nuccore'
        id = ($batch -join ',')
        rettype = 'gb'
        retmode = 'text'
        tool = 'capstone_phage_sae'
    }
    $response = $null
    for ($attempt = 0; $attempt -lt 4 -and $null -eq $response; $attempt++) {
        try {
            $response = Invoke-WebRequest -Uri $baseUri -Method Post -Body $body -UseBasicParsing
        } catch {
            if ($attempt -eq 3) { throw }
            Start-Sleep -Seconds ([int][math]::Pow(2, $attempt))
        }
    }

    $recordText = [string]$response.Content
    $recordCount = [regex]::Matches($recordText, '(?m)^LOCUS\s').Count
    $versions = @([regex]::Matches($recordText, '(?m)^VERSION\s+(\S+)') | ForEach-Object { $_.Groups[1].Value })
    $missingVersions = @($batch | Where-Object { $_ -notin $versions })
    if ($recordCount -ne $batch.Count -or $missingVersions.Count -gt 0) {
        throw "NCBI returned $recordCount records for a batch of $($batch.Count); missing versions: $($missingVersions -join ', ')"
    }

    if (-not $recordText.EndsWith("`n")) { $recordText += "`n" }
    [System.IO.File]::AppendAllText($partial, $recordText, $encoding)
    foreach ($version in $versions) { [void]$found.Add($version) }
    $batchNumber = [math]::Floor($start / $BatchSize) + 1
    Write-Output "Fetched batch $batchNumber/$batchCount ($recordCount records; total $($found.Count))"
    Start-Sleep -Milliseconds 450
}

if ($found.Count -ne $accessions.Count) {
    throw "Downloaded $($found.Count) of $($accessions.Count) expected records. Partial file retained at $partial"
}
[System.IO.File]::Move((Resolve-Path -LiteralPath $partial), $OutputFile)
Write-Output "Saved $($found.Count) records to $OutputFile"
