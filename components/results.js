import { Box } from 'theme-ui'
import { Row, Column } from './'
import { candidateColors, scaleLookup, raceLookup } from './constants'
import { formatPercent, formatDate } from './utils'
import { color } from 'd3-color'
import load from './load'

const Results = ({ selected, setSelectedCandidate, scale, race }) => {
  const { data, error } = load(
    '/results/Mayoral (FAKE DATA).json',
    '/results/City Council 38 (FAKE DATA).json',
  )

  const thisData = data[raceLookup[race]]

  if (!thisData) {
    return <></>
  } else {
    let totals = {}
    let totalReporting
    let thisSelected = selected[scaleLookup[scale]]

    if (
      !thisSelected ||
      !(
        thisData.election_districts[thisSelected] ||
        thisData.assembly_districts[thisSelected]
      )
    ) {
      totals = thisData.all.all.candidates
      totalReporting = thisData.all.all.reporting
    } else {
      if (scale == 'Election district') {
        totals = thisData.election_districts[thisSelected].candidates
        totalReporting = thisData.election_districts[thisSelected].reporting
      }
      if (scale == 'Assembly district') {
        totals = thisData.assembly_districts[thisSelected].candidates
        totalReporting = thisData.assembly_districts[thisSelected].reporting
      }
    }

    const total = Object.values(totals).reduce((a, b) => a + b, 0)
    const sortedTotals = Object.entries(totals).sort((a, b) => b[1] - a[1])

    return (
      <>
        <Box sx={{ px: [4], py: [4] }}>
          <Box
            sx={{
              fontSize: [3, 3, 3, 3],
              fontFamily: 'heading',
              letterSpacing: 'heading',
              display: 'flex',
            }}
          >
            <Box sx={{ textTransform: 'uppercase' }}>Early Results</Box>
            {thisSelected &&
              (scale == 'Election district'
                ? thisData.election_districts[thisSelected]
                : thisData.assembly_districts[thisSelected]) && (
                <Box sx={{ ml: 'auto' }}>
                  {scale == 'Assembly district'
                    ? `AD ${thisSelected}`
                    : `${thisSelected.slice(0, 2)}-${thisSelected.slice(2, 5)}`}
                </Box>
              )}
            {!(
              thisSelected &&
              (scale == 'Election district'
                ? thisData.election_districts[thisSelected]
                : thisData.assembly_districts[thisSelected])
            ) && <Box sx={{ ml: 'auto', textTransform: 'intial' }}>Total</Box>}
          </Box>
          <Box sx={{ mt: [3] }}>
            {sortedTotals.map((k, v) => {
              return (
                <Box key={k} sx={{ mb: [2] }}>
                  <Box sx={{ position: 'relative' }}>
                    <Box
                      sx={{
                        position: 'absolute',
                        height: '6px',
                        zIndex: 1,
                        bottom: '-5px',
                        left: 0,
                        width: `${(100 * k[1]) / total}%`,
                        bg: candidateColors[k[0]],
                      }}
                    />
                    <Box
                      sx={{
                        position: 'absolute',
                        height: '6px',
                        zIndex: 0,
                        bottom: '-5px',
                        left: 0,
                        width: `100%`,
                        bg: 'rgb(220,220,220)',
                      }}
                    />
                    <Box sx={{ display: 'flex' }}>
                      <Box
                        sx={{
                          zIndex: 2,
                          fontSize: [3, 3, 3, 3],
                          cursor: 'pointer'
                        }}
                        onClick={() => setSelectedCandidate(k[0])}
                      >
                        {k[0]}
                      </Box>
                      <Box
                        sx={{
                          ml: 'auto',
                          mt: '4px',
                          fontSize: [1, 1, 1, 1],
                          fontFamily: 'mono',
                          letterSpacing: 'mono',
                          pl: '4px',
                          pr: '4px',
                        }}
                      >
                        {k[1]}
                        <Box sx={{ color: 'muted' }} as='span'>
                          {' '}
                          ({formatPercent(k[1] / total)})
                        </Box>
                      </Box>
                    </Box>
                  </Box>
                </Box>
              )
            })}
          </Box>
          <Box
            sx={{
              zIndex: 2,
              fontSize: [3, 3, 3, 3],
              mt: [5],
            }}
          >
            <Box sx={{ fontFamily: 'mono', fontSize: [1, 1, 1, 1] }}>
              {formatPercent(totalReporting)} REPORTING
            </Box>
            <Box sx={{ fontFamily: 'mono', fontSize: [1, 1, 1, 1] }}>
              LAST UPDATED {formatDate(thisData.last_updated)}
            </Box>
          </Box>
        </Box>
      </>
    )
  }
}

export default Results
