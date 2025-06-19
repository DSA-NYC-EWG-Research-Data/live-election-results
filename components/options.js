import { Box } from 'theme-ui'
import { Select } from './'
import { candidates, candidateColors } from '../components/constants'

const Options = ({
  selectedCandidate,
  setSelectedCandidate,
  scale,
  setScale,
  race,
  setRace,
}) => {
  const allCandidates = candidates[race]

  return (
    <>
      <Box
        sx={{
          borderRadius: [0, '2px', '2px', '2px'],
          bg: 'rgb(255,255,255,0.9)',
          mr: [0, 4, 4, 4],
          mt: [0, 4, 4, 4],
          width: ['calc(100vw)', '400px', '400px', '400px'],
        }}
      >
        <Box sx={{ px: [4], py: [4] }}>
          <Box sx={{}}>
            <Box
              as='label'
              htmlFor='race-select'
              sx={{
                fontSize: [3, 3, 3, 3],
                fontFamily: 'heading',
                letterSpacing: 'heading',
                textTransform: 'uppercase',
              }}
            >
              Select race
            </Box>
            <Select
              id='race-select'
              value={race}
              onChange={(e) => setRace(e.target.value)}
            >
              <option>Mayoral</option>
              <option>City Council 38</option>
            </Select>
          </Box>
        </Box>
      </Box>
      <Box
        sx={{
          borderRadius: [0, '2px', '2px', '2px'],
          bg: 'rgb(255,255,255,0.9)',
          mr: [0, 4, 4, 4],
          mt: [0, 4, 4, 4],
          width: ['calc(100vw)', '400px', '400px', '400px'],
        }}
      >
        <Box sx={{ px: [4], py: [4] }}>
          <Box sx={{ mb: [4] }}>
            <Box
              as='label'
              htmlFor='candidate-select'
              sx={{
                fontSize: [3, 3, 3, 3],
                fontFamily: 'heading',
                letterSpacing: 'heading',
                textTransform: 'uppercase',
              }}
            >
              Filter by Candidate
            </Box>
            <Select
              id='candidate-select'
              value={selectedCandidate}
              onChange={(e) => setSelectedCandidate(e.target.value)}
            >
              <option value='All Candidates'>All Candidates</option>
              {allCandidates.map((candidate) => (
                <option key={candidate} value={candidate}>
                  {candidate}
                </option>
              ))}
            </Select>
          </Box>
          <Box
            as='label'
            htmlFor='scale-select'
            sx={{
              fontSize: [3, 3, 3, 3],
              fontFamily: 'heading',
              letterSpacing: 'heading',
              textTransform: 'uppercase',
            }}
          >
            Show results by
            <Select
              id='scale-select'
              value={scale}
              onChange={(e) => setScale(e.target.value)}
            >
              <option value='Election district'>Election district</option>
              <option value='Assembly district'>Assembly district</option>
            </Select>
          </Box>
        </Box>
      </Box>
    </>
  )
}

export default Options
